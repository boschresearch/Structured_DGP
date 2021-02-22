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

from scipy.linalg import block_diag

from gpflow.kernels import RBF, White
from gpflow.likelihoods import Gaussian

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from structured_dgp.full_dgp import Full_DGP, Fast_Approx_Full_DGP
from structured_dgp.full_dgp import Mean_Field_DGP, Full_DGP_Sampled

def prepare_model(name, X, y, Z, num_samples_train = 5, minibatch = None, M = 30, small_architecture=True):
    """
    Initialize three layer deep GPs with different architectures,
    variational families, and inference methods.
    
    name can be one of {'fc', 'star', 'mf', 'fc_sampled'} and gives
    the fully-coupled, stripes-and-arrow, or mean-field dgp with analytical
    marginalisation of the inducing outputs, or the fully-coupled dgp with
    marginalisation by Monte Carlo sampling, respectively.
    
    The variational parameters are initialized as described in e.g.
    https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/demos/demo_regression_UCI.ipynb
    making the training more effective in the beginning.
    """
    #prepare the kernels (3 layers)
    #use rbf kernels in all layers and additionally white noise kernels in all but the last layer
    #disable training the variance of the rbf kernel in the intermediate layers
    #if small_architecture=True 2 GPs in both hidden layers, otherwise 5    
    dim_X = X.shape[1]
    
    k = RBF(dim_X, ARD=True, lengthscales=1)
    k.variance.set_trainable(False)
    k+=White(dim_X, variance=1e-3)
    
    Ks = [k]
    if small_architecture:
        k = RBF(2, ARD=True, lengthscales=1)
        k.variance.set_trainable(False)
        k+=White(2, variance=1e-3)
        Ks += [k,RBF(2, ARD=True, lengthscales=1)]
    else:
        k = RBF(5, ARD=True, lengthscales=1)
        k.variance.set_trainable(False)
        k+=White(5, variance=1e-3)
        Ks += [k,RBF(5, ARD=True, lengthscales=1)]
    
    assert name in ['fc', 'star', 'mf', 'fc_sampled'], 'Unknown name of dgp model used'
    
    if name == 'fc':
        #fully-coupled
        model = Full_DGP(X, y, Z.copy(), Ks.copy(), Gaussian(0.01),
                         minibatch_size=minibatch, num_samples = num_samples_train)
    elif name == 'star':
        #stripes-and-arrow
        model = Fast_Approx_Full_DGP(X, y, Z.copy(), Ks.copy(), Gaussian(0.01),
                                    stripes = True, arrow = True,minibatch_size=minibatch,
                                    num_samples = num_samples_train)
    elif name == 'mf':
        #mean-field
        model = Mean_Field_DGP(X, y, Z.copy(), Ks.copy(), Gaussian(0.01),
                               minibatch_size=minibatch, num_samples = num_samples_train)
    elif name == 'fc_sampled':
        #fully-coupled with marginalisation by Monte Carlo sampling
        model = Full_DGP_Sampled(X, y, Z.copy(), Ks.copy(), Gaussian(0.01),
                                 minibatch_size=minibatch, num_samples = num_samples_train)
    
    if name in ['fc','fc_sampled']:
        #start the inner layers almost deterministically,
        #this is done by default for mf and star dgp
        SM_prior = model.layers.S_M_sqrt.value
        SM_det = block_diag(SM_prior[0,:-M,:-M]* 1e-5,SM_prior[0,-M:,-M:])
        model.layers.S_M_sqrt = [SM_det]    
    
    return model