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
Test that the sample_first_layer routine is correct by comparing it to the 
salimbeni mf dgp and also to two different cases of self implemented formulas
for GPs.
"""

import numpy as np
import gpflow
import tensorflow as tf
from structured_dgp.all_layers import Fully_Coupled_Layers
from gpflow.likelihoods import Gaussian
from gpflow import autoflow, settings
from doubly_stochastic_dgp.dgp import DGP
from structured_dgp.init_linear import init_linear
import unittest


class autoflow_saldgp(DGP):
            
    @gpflow.params_as_tensors
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]),(gpflow.settings.float_type, [None, None]))
    def get_var(self,X,z=None):
        F = tf.tile(tf.expand_dims(X, 0), [1, 1, 1])
        f, mean, var = self.layers[0].sample_from_conditional(F, z=z, full_cov=False)
        return f, mean, var
    
    @gpflow.params_as_tensors
    def set_qsqrt(self,qsqrt,qmu):
        self.layers[0].q_sqrt = tf.constant(qsqrt)
        self.layers[0].q_mu = tf.constant(qmu)
        
class autoflow_Layer(Fully_Coupled_Layers):
    
    @autoflow((settings.float_type, [None, None]),(settings.int_type, None),(settings.float_type, [None, None]))
    def autoflow_sample(self,X,S,z=None):
        return self.sample_from_first_layer(X,S,z)
    
    @autoflow((settings.float_type, [None, None]))
    def autoflow_mean(self,X):
        return self.all_mean_funcs[0](X)
        
class TestFirstLayer(unittest.TestCase):
    
    """test for block-block diagonal S_M which can be compared to Salimbeni """
    def diag_SM(self):
        X = np.array([[1.,2.,3.],[1.,2.1,3.],[1.1,2.,3.],[1.,2.,3.1]])
        Y = np.array([[1.],[2.],[.2],[3.]])
        Z = np.array([[1.,2.,3.],[1.3,2.2,3.1]])
        z=np.array([[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.]])
        kernels = [gpflow.kernels.RBF(3),gpflow.kernels.RBF(2)]
        sm_sqrt = np.array([[1.,0.,0.,0.,0.,0.],[0.5,1.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.],[0.,0.,0.95,1.,0.,0.],
                       [0.8,0.2,0.35,0.45,1.,0.],[0.15,0.25,0.6,0.9,0.1,1.]])
        
        all_Zs, all_mfs = init_linear(X,Z,kernels)
        mylayers = autoflow_Layer(X,Y,Z,kernels,all_mfs,all_Zs,S_M = sm_sqrt, mu_M = np.array([[1.,2.,3.,4.,5.,6.]]).T)        
        diag_f, _,_,_,_,_ = mylayers.autoflow_sample(X,1,z)
        diag_f = diag_f[:,:2]
        
        Saldgp = autoflow_saldgp(X,Y,Z,kernels,Gaussian())
        myqsqrt = np.array([[[1.,0.],[0.5,1.]],[[1.,0.],[0.95,1.]]])
        myqmu = np.array([[1.,3.],[2.,4.]])
        Saldgp.set_qsqrt(myqsqrt,myqmu)
        sal_f, _, _ = Saldgp.get_var(X,z)
        sal_f = sal_f[0]        
        
        return sal_f, diag_f
        
    """ test for constant dimensions"""
    def full_SM(self):
        def K(X1, X2):
            
            X1, X2 = np.array(X1), np.array(X2)
            if X1.ndim == 1:
                X1 = np.array([X1])
            if X2.ndim == 1:
                X2 = np.array([X2])
            
            if np.size(X1,0) != np.size(X2,0):
                print("Size of input matrices don't match")
                sys.exit()
            
            m1, m2 = np.size(X1,1), np.size(X2,1)
                
            matK = np.zeros((m1,m2))
            
            for i in range(m1):
                for j in range(m2):
                    matK[i, j] = np.exp(-1/2 * (X1[:,i] - X2[:,j]).dot(X1[:,i] - X2[:,j]))
                        
            return matK
        
        X = np.array([[1.,2.],[1.,2.1],[1.3,2.],[1.2,2.4]])
        Y = X.copy()
        Z = X.copy()
        A = np.tril(np.random.rand(16,16)) #"cholesky" of S_M 
        B = np.random.rand(16,1)
        z=np.array([[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.]])
        kernels = [gpflow.kernels.RBF(2),gpflow.kernels.RBF(2,lengthscales=2.)]
        all_Zs, all_mfs = init_linear(X,Z,kernels)
        mylayers = autoflow_Layer(X,Y,Z,kernels,all_mfs,all_Zs,S_M = A, mu_M = B)        
        full_f, _,_,_,_,_ = mylayers.autoflow_sample(X,1,z)
        
        full_test_f = []
        sm = A @ A.T
        for k in range(4):
            knm = np.kron(np.eye(2),np.matmul(K(X.T,Z.T)[k],np.linalg.inv(K(Z.T,Z.T))))
            smtilde = np.matmul(np.matmul(knm, sm[:8,:8]),knm.T)
            smtilde += np.kron(np.eye(2),1-np.matmul(np.matmul(K(X.T,Z.T)[k],np.linalg.inv(K(Z.T,Z.T))),K(X.T,Z.T)[k].T))
            l = np.linalg.cholesky(smtilde)
            full_test_f.append((np.matmul(knm,B[:8]) + np.matmul(l,np.array([z[k]]).T)).T)
            
        return full_f, np.array(full_test_f)[:,0,:] + X #+X due to linear mean function
    
    """ test for changing dimensions"""
    def full_SM2(self):
        def K(X1, X2):
            
            X1, X2 = np.array(X1), np.array(X2)
            if X1.ndim == 1:
                X1 = np.array([X1])
            if X2.ndim == 1:
                X2 = np.array([X2])
            
            if np.size(X1,0) != np.size(X2,0):
                print("Sizes of input matrices don't match")
                sys.exit()
            
            m1, m2 = np.size(X1,1), np.size(X2,1)
                
            matK = np.zeros((m1,m2))
            
            for i in range(m1):
                for j in range(m2):
                    matK[i, j] = np.exp(-1/2 * (X1[:,i] - X2[:,j]).dot(X1[:,i] - X2[:,j]))
                        
            return matK
        
        
        X = np.array([[1.,2.,3.],[1.,2.1,3.],[1.1,2.,3.],[1.,2.,3.1]])
        Y = np.array([[1.],[2.],[.2],[3.]])
        Z = np.array([[1.,2.,3.],[1.3,2.2,3.1]])
        z=np.array([[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.]])
        kernels = [gpflow.kernels.RBF(3),gpflow.kernels.RBF(2)]
        sm_sqrt = np.array([[1.,0.,0.,0.,0.,0.],[0.5,1.,0.,0.,0.,0.],[0.1,0.3,1.,0.,0.,0.],[0.7,0.4,0.95,1.,0.,0.],
                       [0.8,0.2,0.35,0.45,1.,0.],[0.15,0.25,0.6,0.9,0.1,1.]])
        sm = np.matmul(sm_sqrt,sm_sqrt.T)
        
        all_Zs, all_mfs = init_linear(X,Z,kernels)
        mylayers = autoflow_Layer(X,Y,Z,kernels,all_mfs,all_Zs,S_M = sm_sqrt, mu_M = np.array([[1.,2.,3.,4.,5.,6.]]).T)        
        full_f, _,_,_,_,_ = mylayers.autoflow_sample(X,1,z)
        
        full_test_f = []
        for k in range(4):
            knm = np.kron(np.eye(2),np.matmul(K(X.T,Z.T)[k],np.linalg.inv(K(Z.T,Z.T))))
            smtilde = np.matmul(np.matmul(knm, sm[:4,:4]),knm.T)
            smtilde += np.kron(np.eye(2),1-np.matmul(np.matmul(K(X.T,Z.T)[k],np.linalg.inv(K(Z.T,Z.T))),K(X.T,Z.T)[k].T))
            l = np.linalg.cholesky(smtilde)
            full_test_f.append((np.matmul(knm,np.array([[1,2,3,4]]).T) + np.matmul(l,np.array([z[k]]).T)).T)
        
        return full_f, np.array(full_test_f)[:,0,:] + mylayers.autoflow_mean(X) #+X due to linear mean function
    
      
    def test_first_layer_diag(self):
        sal_f, diag_f = self.diag_SM()
        self.assertTrue(np.allclose(sal_f,diag_f,atol=1e-03))
        
    def test_first_layer_full(self):
        full_f, full_test_f = self.full_SM()
        self.assertTrue(np.allclose(full_f,full_test_f,atol=1e-03))
    
    def test_first_layer_full2(self):
        full_f, full_test_f = self.full_SM2()
        self.assertTrue(np.allclose(full_f,full_test_f,atol=1e-03))  

    
if __name__ == '__main__':
    unittest.main()