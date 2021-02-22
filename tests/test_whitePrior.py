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
Test the white prior. Explicit description below.
"""

import numpy as np
from scipy.linalg import block_diag
import gpflow
from structured_dgp.full_dgp import Full_DGP
from gpflow.likelihoods import Gaussian
from gpflow import settings
float_type = settings.float_type

import unittest

class my_FullDGP(Full_DGP):

    @gpflow.params_as_tensors
    @gpflow.autoflow()
    def get_KL(self):
        KL = self.layers.KL()
        return KL
    
    @gpflow.params_as_tensors
    @gpflow.autoflow()
    def get_LM(self):
        self.layers.build_cholesky_if_needed()        
        LMM = self.layers.all_LMM
        return LMM
      
class TestWhitening(unittest.TestCase):
    """Test that a model decribed by q(f_M)= N(f_M | mu_M, S_M) and p(f_M) = N(f_M | 0, L_K @ L_K^T)
    has the same KL and ELBO as a model with a whitened prior and a  differently parameterised variational
    posterior, i.e. q(f_M)= N(f_M | L_K_inv @ mu_M, L_K_inv @ S_M @ L_K_inv^T) and p(f_M) = N(f_M | 0, I)
    """
    
    def get_ELBOs(self):

        X = np.array([[1.,2.,3.],[1.4,2.1,3.],[1.1,2.,3.],[1.,2.,3.1]])
        Y = np.array([[0.],[2.],[1.2],[3.5]])
        Z = np.array([[1.,2.,3.5],[1.3,2.2,3.1]])        
        sm_sqrt = np.tril(np.random.rand(8,8))
        mu_M = np.array([[1.,2.8,3.,4.,5.7,6.,3.,3.2]]).T
        
        kernels = [gpflow.kernels.RBF(3),gpflow.kernels.RBF(2,lengthscales=4.0),gpflow.kernels.RBF(1,lengthscales=2.0)]
        mydgp = my_FullDGP(X.copy(),Y.copy(),Z.copy(),kernels,Gaussian(),mu_M=mu_M,S_M=sm_sqrt,whitened_prior=False)
        zs =[[[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.2]],[[.1],[.2],[1.2],[0.1]],[[1.],[.5],[.25],[0.5]]]
        ELBO_normal = mydgp._build_likelihood(zs)
        sess = mydgp.enquire_session()
        ELBO_normal = sess.run(ELBO_normal)
        KL_normal = mydgp.get_KL()
        
        LMMs = mydgp.get_LM()
        L_K_inv = np.linalg.inv(block_diag(LMMs[0],LMMs[0],LMMs[1],LMMs[2]))
        mu_M2, sm_sqrt2 = L_K_inv @ mu_M, L_K_inv @ sm_sqrt
        kernels2 = [gpflow.kernels.RBF(3),gpflow.kernels.RBF(2,lengthscales=4.0),gpflow.kernels.RBF(1,lengthscales=2.0)]
        mydgp2 = my_FullDGP(X,Y,Z,kernels2,Gaussian(),mu_M=mu_M2,S_M=sm_sqrt2,whitened_prior=True)
        ELBO_white = mydgp2._build_likelihood(zs)
        sess2 = mydgp2.enquire_session()
        ELBO_white = sess2.run(ELBO_white)
        KL_white = mydgp2.get_KL()        
        
        return ELBO_normal, KL_normal, ELBO_white, KL_white
        
    def test_diag_ELBO(self):
        ELBO_normal, KL_normal, ELBO_white, KL_white = self.get_ELBOs()
        self.assertAlmostEqual(ELBO_normal/ELBO_white, 1,places=4)
        self.assertAlmostEqual(KL_normal/KL_white, 1,places=4)

    
if __name__ == '__main__':
    unittest.main()