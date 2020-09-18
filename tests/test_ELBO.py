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

"""
Test comparing ELBOs and KLs of the mf dgp with the fc dgp for special cases,
once with a whitened prior and once without.
"""

import numpy as np
import gpflow
import tensorflow as tf
from structured_dgp.full_dgp import Full_DGP
from gpflow.likelihoods import Gaussian
from doubly_stochastic_dgp.dgp import DGP
import unittest

class sal0dgp(DGP):
    
    @gpflow.params_as_tensors
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]),(gpflow.settings.float_type, [None, None]))
    def get_var(self,X,z=None):
        F = tf.tile(tf.expand_dims(X, 0), [1, 1, 1])
        f, mean, var = self.layers[0].sample_from_conditional(F, z=z, full_cov=False)
        return f, mean, var
    
    @gpflow.params_as_tensors
    def set_qsqrt(self,qsqrt,qmu):
        for j in range(3):
            self.layers[j].q_sqrt = tf.constant(qsqrt[j],dtype =gpflow.settings.float_type)
            self.layers[j].q_mu = tf.constant(qmu[j],dtype =gpflow.settings.float_type)   
            
    @gpflow.params_as_tensors
    def my_propagate(self, X, z1,z2,z3):
        F = tf.tile(tf.expand_dims(X, 0), [1, 1, 1])

        Fs, Fmeans, Fvars = [], [], []
        F, Fmean, Fvar = self.layers[0].sample_from_conditional(F, z=z1, full_cov=False)
        Fs.append(F)
        Fmeans.append(Fmean)
        Fvars.append(Fvar)
        F, Fmean, Fvar = self.layers[1].sample_from_conditional(F, z=z2, full_cov=False)
        Fs.append(F)
        Fmeans.append(Fmean)
        Fvars.append(Fvar)
        F, Fmean, Fvar = self.layers[2].sample_from_conditional(F, z=z3, full_cov=False)
        Fs.append(F)
        Fmeans.append(Fmean)
        Fvars.append(Fvar)
        return Fs, Fmeans, Fvars
    
    @gpflow.params_as_tensors
    def my_build_predict(self, X,z1,z2,z3):
        Fs, Fmeans, Fvars = self.my_propagate(X, z1,z2,z3)
        return Fmeans[-1], Fvars[-1]
    
    @gpflow.params_as_tensors
    def myE_log_p_Y(self, X, Y, z1, z2, z3):
        """
        Calculate the expectation of the data log likelihood under the variational distribution
         with MC samples
        """
        Fmean, Fvar = self.my_build_predict(X, z1, z2, z3)
        var_exp = self.likelihood.variational_expectations(Fmean, Fvar, Y)  # S, N, D
        var_exp = tf.reduce_mean(var_exp, 0)  # N, D
        return var_exp
    
    @gpflow.params_as_tensors
    @gpflow.autoflow()
    def get_KL(self):
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        return KL
    
    @gpflow.params_as_tensors
    def my_build_likelihood(self,z1,z2,z3):
        L = tf.reduce_sum(self.myE_log_p_Y(self.X, self.Y, z1, z2, z3))
        KL = tf.reduce_sum([layer.KL() for layer in self.layers])
        scale = tf.cast(self.num_data, gpflow.settings.float_type)
        scale /= tf.cast(tf.shape(self.X)[0], gpflow.settings.float_type)  # minibatch size
        return L * scale - KL
    
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]),(gpflow.settings.float_type, [None, None]),
                     (gpflow.settings.float_type, [None, None]))
    def my_ELBO(self,z1,z2,z3):
        return self.my_build_likelihood(z1,z2,z3)

class my_FullDGP(Full_DGP):

    @gpflow.params_as_tensors
    @gpflow.autoflow()
    def get_KL(self):
        KL = self.layers.KL()
        return KL
      
class TestELBO(unittest.TestCase):
    
    def get_Full_and_MF_ELBOs(self,white):

        X = np.array([[1.,2.,3.],[1.,2.1,3.],[1.1,2.,3.],[1.,2.,3.1]])
        Y = np.array([[1.],[2.],[1.2],[3.]])
        Z = np.array([[1.,2.,3.],[1.3,2.2,3.1]])        
        sm_sqrt2 = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],[0.5,1.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.],
                             [0.,0.,0.95,1.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.1,1.,0.,0.],
                             [0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.25,1.]])
        mu_M = np.array([[1.,2.,3.,4.,5.,6.,3.,3.]]).T
        
        kernels = [gpflow.kernels.RBF(3),gpflow.kernels.RBF(2,lengthscales=4.0),gpflow.kernels.RBF(1,lengthscales=2.0)]
        #all_Zs, all_mean_funcs = init_linear(X,Z,kernels)
        #mylayers = Fully_Coupled_Layers(X,Y,Z,kernels,all_mean_funcs,all_Zs,S_M = sm_sqrt2, mu_M = mu_M)
        mydgp = my_FullDGP(X,Y,Z,kernels,Gaussian(),mu_M=mu_M,S_M=sm_sqrt2,whitened_prior=white)#,mylayers)
        zs=[[[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.]],[[.1],[.2],[.2],[0.1]],[[1.],[.5],[.2],[0.5]]]
        
        #f, muNtilde, SNtilde, K1NM, mean, var = mydgp.propagate(X,zs=zs)
        ELBO_diag = mydgp._build_likelihood(zs=zs)
        session = gpflow.get_default_session()
        ELBO_diag = session.run(ELBO_diag)
        
        z1 = [[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.]]
        z2 = [[.1],[.2],[.2],[0.1]]
        z3 = [[1.],[.5],[.2],[0.5]]
        Saldgp = sal0dgp(X,Y,Z,kernels,Gaussian(),white=white)
        myqsqrt = np.array([[[[1.,0.],[0.5,1.]],[[1.,0.],[0.95,1.]]],[[[1.,0.],[0.1,1.]]],[[[1.,0.],[0.25,1.]]]])
        myqmu = [[[1.,3.],[2.,4.]],[[5.],[6.]],[[3.],[3.]]]
        Saldgp.set_qsqrt(myqsqrt,myqmu)
        ELBO_sal = Saldgp.my_ELBO(z1,z2,z3)
        
        return ELBO_diag, ELBO_sal, mydgp.get_KL(), Saldgp.get_KL()
        
    def test_diag_ELBO_white_prior(self):
        ELBO_diag, ELBO_sal, KL_diag, KL_sal = self.get_Full_and_MF_ELBOs(True)
        self.assertAlmostEqual(ELBO_diag/ELBO_sal, 1,places=4)
        self.assertAlmostEqual(KL_diag/KL_sal, 1,places=4)
    
    def test_diag_ELBO_normal_prior(self):
        ELBO_diag, ELBO_sal, KL_diag, KL_sal = self.get_Full_and_MF_ELBOs(False)
        self.assertAlmostEqual(ELBO_diag/ELBO_sal, 1,places=4)
        self.assertAlmostEqual(KL_diag/KL_sal, 1,places=4)

    
if __name__ == '__main__':
    unittest.main()