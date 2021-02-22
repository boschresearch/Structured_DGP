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
Tests for the STAR DGP. For different settings of white prior, and the stripes
and arrow flags of the STAR DGP, the ELBOs and KLs of the STAR DGP and the
(equivalent) FC DGP with the restricted covariance matrix are compared.
"""

import numpy as np
import gpflow
from structured_dgp.full_dgp import Full_DGP,Fast_Approx_Full_DGP
from gpflow.likelihoods import Gaussian
from gpflow.kernels import RBF
import unittest

class my_FullDGP(Full_DGP):

    @gpflow.params_as_tensors
    @gpflow.autoflow()
    def get_KL(self):
        KL = self.layers.KL()
        return KL
    
class my_ApproxDGP(Fast_Approx_Full_DGP):

    @gpflow.params_as_tensors
    @gpflow.autoflow()
    def get_KL(self):
        KL = self.layers.KL()
        return KL

    def set_blocks(self,arrow = True, stripes = True):    
        if arrow:
            self.layers.arrow_blocks = np.random.rand(6,50,50)
        if stripes:
            self.layers.stripe_blocks = np.random.rand(3,50,50)
        self.layers.diag_blocks = np.random.rand(7,50,50)
        return self.layers.S_M_sqrt_value()[0]
     
class TestELBO(unittest.TestCase):
    
    def get_Full_and_Approx_ELBOs(self,white,arrow,stripes):
        
        X = np.random.randn(300,3)
        Y = np.random.randn(300,1)
        Z = np.random.randn(50,3)
        mu_M = np.random.randn(350,1)
        
        kernels = [RBF(3),RBF(3,lengthscales=4.0),RBF(3,lengthscales=2.0)]
        approx_dgp = my_ApproxDGP(X,Y,Z,kernels,Gaussian(),mu_M=mu_M,
                                  whitened_prior=white,arrow=arrow,stripes=stripes)
        SM = approx_dgp.set_blocks(arrow,stripes)
        mydgp = my_FullDGP(X,Y,Z,kernels,Gaussian(),mu_M=mu_M,S_M = SM,whitened_prior=white)
        zs = [np.random.randn(300,3),np.random.randn(300,3),np.random.randn(300,1)]
        
        ELBO_full = mydgp._build_likelihood(zs=zs)
        session = mydgp.enquire_session()
        ELBO_full = session.run(ELBO_full)
        
        ELBO_approx = approx_dgp._build_likelihood(zs=zs)
        session = approx_dgp.enquire_session()
        ELBO_approx = session.run(ELBO_approx)
        
        return ELBO_full, ELBO_approx, mydgp.get_KL(), approx_dgp.get_KL()
    """All possible combinations of white, arrow, stripes follow """  
    def test_white_arrow_stripes(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(True,True,True)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)
    
    def test_white_arrow(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(True,True,False)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)
    
    def test_white_stripes(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(True,False,True)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)
    
    def test_white(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(True,False,False)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)
    
    def test_arrow_stripes(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(False,True,True)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)
    
    def test_arrow(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(False,True,False)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)
    
    def test_stripes(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(False,False,True)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)
    
    def test_nothing(self):
        ELBO_full, ELBO_approx,KL_full, KL_approx = self.get_Full_and_Approx_ELBOs(False,False,False)
        self.assertAlmostEqual(ELBO_full/ELBO_approx, 1,places=4)
        self.assertAlmostEqual(KL_full/KL_approx, 1,places=4)

    
if __name__ == '__main__':
    unittest.main()