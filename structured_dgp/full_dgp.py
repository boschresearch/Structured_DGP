# -*- coding: utf-8 -*-
# Copyright (c) 2020 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Jakob Lindinger, jakob.lindinger@de.bosch.com


import numpy as np
import tensorflow as tf

from gpflow.params import DataHolder, Minibatch
from gpflow import autoflow, params_as_tensors
from gpflow.models.model import Model
from gpflow import settings
float_type = settings.float_type

from structured_dgp.all_layers import Fully_Coupled_Layers, Stripes_Arrow_Layers, Fast_Stripes_Arrow_Layers
from structured_dgp.init_linear import init_linear

"""
The following classes are adapted from Doubly-Stochastic-DGP V 1.0
( https://github.com/ICL-SML/Doubly-Stochastic-DGP/ 
Copyright 2017 Hugh Salimbeni, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree)
to incorporate couplings between different latent GPs.
"""

class Full_DGP_Base(Model):
    def __init__(self, X, Y, likelihood, layers, minibatch_size=None,
                 num_samples=1, num_data=None, **kwargs):
        """
        Base class for the fully coupled DGP providing all basic functionalities.
        """
        Model.__init__(self, **kwargs)
        self.num_samples = num_samples

        self.num_data = num_data or X.shape[0]
        
        if minibatch_size:
            self.X = Minibatch(X, minibatch_size, seed=0)
            self.Y = Minibatch(Y, minibatch_size, seed=0)
        else:
            self.X = DataHolder(X)
            self.Y = DataHolder(Y)

        self.likelihood = likelihood
        self.layers = layers
        
        
    def propagate(self, X, S=1, zs=None):
        """
        propagate input points X  S times through the deep GP.
        zs (a list of numpy arrays with L elements each of size (S*N,T_l) respectively)
        can be provided to be used for reproducibility in the reparametrization trick.
        returns samples drawn along the way and also means and variances and other quantities which are
        needed in the full DGP for sampling.
        returns stacks of outputs, i.e. always S samples for every point
        """        
        if zs is None:
            zs = [None, ] * self.layers.all_Gps_per_layer.shape[0]
        else:
            zs = [np.array(z) for z in zs]
            
        fN, mean, var, muNtilde, SNtilde_chol, KMN_inv = self.layers.sample_from_first_layer(X,S,z=zs[0])
        Fmeans, Fvars = [mean],[var]
        for l in range(self.layers.all_Gps_per_layer.shape[0] - 1 ):
            z = zs[l+1]
            fN,mean,var,muNtilde,SNtilde_chol,KMN_inv = self.layers.sample_from_lth_layer(l+1,fN,muNtilde,
                                                                                          SNtilde_chol,
                                                                                          KMN_inv,z=z)
            Fmeans.append(mean)
            Fvars.append(var)

        return fN, muNtilde, SNtilde_chol, KMN_inv, Fmeans, Fvars
    
    def _build_predict(self, X, S=1, zs=None):
        """
        return S stacks of means and variances of the final layer Gaussian
        when data points X are propagated S times through DGP by MC sampling 
        """
        _,_,_,_, SFmeans, SFvars = self.propagate(X,S=S,zs=zs)
        return SFmeans[-1], SFvars[-1] #S*N,T_L and S*N,T_L,T_L

    def E_log_p_Y(self, X, Y, zs = None):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
        with MC samples from propagating the data X num_samples times through the DGP
        """
        S=self.num_samples
        SFmean_tiled, SFvar_tiled = self._build_predict(X,S=S,zs=zs)
        N,T_L = tf.shape(X)[0], tf.shape(Y)[1]
        SFmean,SFvar = tf.reshape(SFmean_tiled,[S,N,T_L]),tf.reshape(SFvar_tiled,[S,N,T_L,T_L ])
        var_exp = self.likelihood.variational_expectations(SFmean, SFvar[:,:,0,:], Y)  #S, N, T_L
        
        return tf.reduce_mean(var_exp, 0)  # N, T_L
    
    @params_as_tensors
    def _build_likelihood(self,zs=None):
        """
        ELBO (optimization objective)
        """        
        data_fit_term = tf.reduce_sum(self.E_log_p_Y(self.X, self.Y,zs))
        KL_term = self.layers.KL()
        #data_fit_term has to be scaled to account for the mini_batch size
        scale = tf.cast(self.num_data, float_type)
        scale /= tf.cast(tf.shape(self.X)[0], float_type)  # minibatch size
        return data_fit_term * scale - KL_term
    
    @autoflow((float_type, [None, None]), (tf.int32, []))
    def predict_y(self, Xnew, num_samples):
        """
        Predict output values at new input points Xnew by sampling num_samples
        times through the DGP
        """
        S = num_samples
        N,T_L = tf.shape(Xnew)[0], self.layers.all_Gps_per_layer[-1]
        Fmean_tiled, Fvar_tiled = self._build_predict(Xnew, S=S)
        Fmean,Fvar = tf.reshape(Fmean_tiled,[S,N,T_L]),tf.reshape(Fvar_tiled,[S,N,T_L,T_L ])
        return self.likelihood.predict_mean_and_var(Fmean, Fvar[:,:,0,:])
    
class Full_DGP(Full_DGP_Base):
    """
    This is the  fully coupled Doubly-Stochastic Deep GP
    with linear/identity mean functions at each layer.
    """
    def __init__(self, X, Y, Z, kernels, likelihood,mu_M=None,S_M=None,
                 whitened_prior=False,initialized_Zs=False,**kwargs):
        """
        X, (N,D) input points
        Y, (N,1) output points
        Z, (M,D) inducing points
        kernels, a list of kernels (one for every layer)
        likelihood, lieklihood object from gpflow
        
        optional initializations:
        mu_M, (M*T,1) the means of the varational distribution q(f_M)
        S_M, (M*T,M*T) actually rather S_M_sqrt, the lower traingular Cholesky factor of S_M,
                       the covariance matrix of q(f_M)
        
        whitened_prior, flag for whitening the prior
        initialized_Zs, flag needed for passing initialized Z values (for all layers)
                        in this case Z needs to be a list of all inducing points of all layers
        """
        #mean functions need to be initialized first
        all_Zs, all_mfs = init_linear(X, Z, kernels,initialized_Zs=initialized_Zs) 
        layers = Fully_Coupled_Layers(X,Y,Z, kernels,all_mfs,all_Zs, mu_M=mu_M,S_M=S_M,
                                      whitened_prior=whitened_prior)
        Full_DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)


class Approx_Full_DGP(Full_DGP_Base):
    """ 
    Naive implementation of the STAR dgp, simply implements the covariance matrix
    and otherwise uses all functionalities of the Full_DGP.
    
    Can be used as a blueprint to naively implement other possible covariance matrices.
    """
    def __init__(self, X, Y, Z, kernels, likelihood,mu_M=None,S_M=None,whitened_prior=False,
                 initialized_Zs=False,arrow=True, stripes=True,**kwargs):
        """
        additional inputs:
        arrow, flag for including the arrowheads of the STAR covariance matrix
        stripes, flag for including the off-diagonal stripes of the STAR covariance matrix
        
        note that if both flags are False, the mean-field dgp from
        https://github.com/ICL-SML/Doubly-Stochastic-DGP/ is reproduced
        """
        all_Zs, all_mfs = init_linear(X, Z, kernels,initialized_Zs=initialized_Zs)
        layers = Stripes_Arrow_Layers(X,Y,Z, kernels,all_mfs,all_Zs, mu_M=mu_M,S_M=S_M,
                                      whitened_prior=whitened_prior, arrow=arrow,stripes=stripes)
        Full_DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)

class Fast_Approx_Full_DGP(Full_DGP_Base):
    """
    Fast implementation of the STAR dgp, that actually exploits the structure
    and sparsity of the covariance matrix.
    """
    def __init__(self, X, Y, Z, kernels, likelihood,mu_M=None,S_M=None,whitened_prior=False,
                 initialized_Zs=False,arrow=True, stripes=True,**kwargs):
        all_Zs, all_mfs = init_linear(X, Z, kernels,initialized_Zs=initialized_Zs)
        layers = Fast_Stripes_Arrow_Layers(X,Y,Z, kernels,all_mfs,all_Zs, mu_M=mu_M,S_M=S_M,
                                      whitened_prior=whitened_prior, arrow=arrow,stripes=stripes)
        Full_DGP_Base.__init__(self, X, Y, likelihood, layers, **kwargs)