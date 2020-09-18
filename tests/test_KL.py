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
Two test comparing the implementation of the KL with an implementation of the
KL for Gaussian distribution from GPflow.
"""

import numpy as np
from gpflow.kullback_leiblers import gauss_kl
from gpflow import kernels, settings, get_default_session
import tensorflow as tf
from structured_dgp.all_layers import Fully_Coupled_Layers
from structured_dgp.init_linear import init_linear
import unittest
import scipy


class TestKL(unittest.TestCase):
    """first test for constant dimensions"""
    def myKL(self):
        X = np.array([[1.,2.],[1.,2.1],[1.3,2.],[1.2,2.4]])
        Y = X.copy()
        Z = X.copy()
        A = np.tril(np.random.rand(16,16)) #"cholesky" of S_M 
        B = np.random.rand(16,1) #mu_M
        all_kernels = [kernels.RBF(2),kernels.RBF(2,lengthscales = 3.,variance = 2.)]
        all_Zs, all_mfs = init_linear(X,Z,all_kernels)
        mylayers = Fully_Coupled_Layers(X,Y,Z,all_kernels,all_mfs,all_Zs, mu_M = B, S_M = A)
        kl = mylayers.KL()
        session = get_default_session()
        kl = session.run(kl)
        
        Kmm1 = all_kernels[0].compute_K_symm(Z) + np.eye(Z.shape[0]) * settings.jitter
        Kmm2 = all_kernels[1].compute_K_symm(Z) + np.eye(Z.shape[0]) * settings.jitter
        K_big = scipy.linalg.block_diag(Kmm1,Kmm1,Kmm2,Kmm2)
        tfKL = gauss_kl(tf.constant(B), tf.constant(A[np.newaxis]), K=tf.constant(K_big))
        
        sess = tf.Session()
        return kl, sess.run(tfKL)

    def test_kl(self):
        kl1, kl2 = self.myKL()
        self.assertAlmostEqual(kl1,kl2,places=3) #only almost eqal due to jitter rounding errors
        
    """second  test for changing dimensions"""
    def myKL2(self):
        X = np.array([[1.,2.,3.],[1.,2.1,3.],[1.1,2.,3.],[1.,2.,3.1]])
        Y = np.array([[1.],[2.],[.2],[3.]])
        Z = np.array([[1.,2.,3.],[1.3,2.2,3.1]])
        A = np.tril(np.random.rand(6,6)) #"cholesky" of S_M 
        B = np.random.rand(6,1) #mu_M
        all_kernels = [kernels.RBF(3),kernels.RBF(2,lengthscales = 3.,variance = 2.)]
        all_Zs, all_mfs = init_linear(X,Z,all_kernels)
        mylayers = Fully_Coupled_Layers(X,Y,Z,all_kernels,all_mfs,all_Zs, mu_M = B, S_M = A)
        kl = mylayers.KL()
        session = get_default_session()
        kl = session.run(kl)
        
        Kmm1 = all_kernels[0].compute_K_symm(all_Zs[0]) + np.eye(Z.shape[0]) * settings.jitter
        Kmm2 = all_kernels[1].compute_K_symm(all_Zs[1]) + np.eye(all_Zs[1].shape[0]) * settings.jitter
        K_big = scipy.linalg.block_diag(Kmm1,Kmm1,Kmm2)
        tfKL = gauss_kl(tf.constant(B), tf.constant(A[np.newaxis]), K=tf.constant(K_big))
        
        sess = tf.Session()
        return kl, sess.run(tfKL)

    def test_kl2(self):
        kl1, kl2 = self.myKL2()
        self.assertAlmostEqual(kl1,kl2,places=3) #only almost eqal due to jitter rounding errors
if __name__ == '__main__':
    unittest.main()
