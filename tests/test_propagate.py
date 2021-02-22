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
Test for the propagate subroutine, first that the results for a restricted
covariance matrix are the same as for the mf dgp, second that a special
form of the covariance matrix leads to an analytically calculated result.
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
    @gpflow.autoflow((gpflow.settings.float_type, [None, None]),(gpflow.settings.float_type, [None, None]),
                     (gpflow.settings.float_type, [None, None]),(gpflow.settings.float_type, [None, None]))
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
        
class TestPropagate(unittest.TestCase):
    
    def diag_SM(self):
        X = np.array([[1.,2.,3.],[1.,2.1,3.],[1.1,2.,3.],[1.,2.,3.1]])
        Y = np.array([[1.],[2.],[.2],[3.]])
        Z = np.array([[1.,2.,3.],[1.3,2.2,3.1]])
        kernels = [gpflow.kernels.RBF(3),gpflow.kernels.RBF(2,lengthscales=4.0),gpflow.kernels.RBF(1,lengthscales=2.0)]
        sm_sqrt2 = np.array([[1.,0.,0.,0.,0.,0.,0.,0.],[0.5,1.,0.,0.,0.,0.,0.,0.],[0.,0.,1.,0.,0.,0.,0.,0.],
                             [0.,0.,0.95,1.,0.,0.,0.,0.],[0.,0.,0.,0.,1.,0.,0.,0.],[0.,0.,0.,0.,0.1,1.,0.,0.],
                             [0.,0.,0.,0.,0.,0.,1.,0.],[0.,0.,0.,0.,0.,0.,0.25,1.]])
        mu_M = np.array([[1.,2.,3.,4.,5.,6.,3.,3.]]).T
        
        mydgp = Full_DGP(X,Y,Z,kernels,Gaussian(),S_M = sm_sqrt2, mu_M = mu_M)
        zs=[[[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.]],[[.1],[.2],[.2],[0.1]],[[1.],[.5],[.2],[0.5]]]
        
        diag_f, _,_,_,_,_ = mydgp.propagate(X,zs=zs)
        session = gpflow.get_default_session()
        diag_f = session.run(diag_f)
        z1 = [[0.1,0.5],[-0.3,0.2],[1.,-1.3],[2.,0.]]
        z2 = [[.1],[.2],[.2],[0.1]]
        z3 = [[1.],[.5],[.2],[0.5]]
        Saldgp = sal0dgp(X,Y,Z,kernels,Gaussian())
        myqsqrt = np.array([[[[1.,0.],[0.5,1.]],[[1.,0.],[0.95,1.]]],[[[1.,0.],[0.1,1.]]],[[[1.,0.],[0.25,1.]]]])
        myqmu = [[[1.,3.],[2.,4.]],[[5.],[6.]],[[3.],[3.]]]
        Saldgp.set_qsqrt(myqsqrt,myqmu)
        temp, _, _ = Saldgp.my_propagate(X,z1,z2,z3)
        sal_f = temp[0][0]
        sal_f = np.append(sal_f,temp[1][0],axis=1)
        sal_f = np.append(sal_f,temp[2][0],axis=1)
        return sal_f, diag_f
    
    def full_SM(self):
        X = np.array([[1.,2.],[1.,2.1],[1.3,2.],[1.2,2.4]])
        Y = X.copy()
        Z = X.copy()
        kernels = [gpflow.kernels.RBF(2),gpflow.kernels.RBF(2,lengthscales = 3.,variance = 2.)]
        np.random.seed(2)
        a = np.random.rand(8,8)
        A = np.tril(a)
        out = np.zeros((4,2,2))
        for k in range(4):
            KMn =  np.kron(np.eye(2),np.matmul(np.linalg.inv(kernels[0].compute_K_symm(X)),kernels[0].compute_K(X, [X[k]])))
            K2MM = np.kron(np.eye(2),kernels[1].compute_K_symm(X))
            
            B = np.linalg.cholesky(K2MM)
            S11 = np.matmul(A,A.T) 
            S12 = np.matmul(A,B.T)
            temp1 = np.matmul(S12.T,KMn)
            temp2 = np.matmul(np.matmul(KMn.T,S11),KMn)
            S22 = np.matmul(np.matmul(temp1,np.linalg.inv(temp2)),temp1.T) + K2MM + 0.0000001* np.eye(K2MM.shape[0])
                        
            S_M = np.block([[S11,S12],[S12.T,S22]])
            S_M_sqrt = np.linalg.cholesky(S_M)
            mu_M = np.array([[1.,2.,3.,4.,5.,6.,3.,3.,1.,2.,3.,4.,5.,6.,3.,3.]]).T
            mydgp = Full_DGP(X,Y,Z,kernels,Gaussian(),S_M = S_M_sqrt, mu_M = mu_M)
            _,_,_,_,_, var = mydgp.propagate(X)
            session = gpflow.get_default_session()
            var = session.run(var)
            var = np.array(var)
            out[k] = var[1,k]        
        return out
    
    """ Test that the algorithm does the same for a block-block diagonal form of the covariance matrix as salimbeni"""   
    def test_propagate_diag(self):
        sal_f, diag_f = self.diag_SM()
        self.assertTrue(np.allclose(sal_f,diag_f,rtol=1e-03))
        
    """ Test that for a special setting of the full covariance matrix for the inducing points,
        the propagation results in a special covariance matrix for the data points drawn after the second layer"""    
    def test_propagate_full(self):
        out = self.full_SM()
        out_test = np.tile(2*np.eye(2),(4,1,1))
        print(out,out_test)
        self.assertTrue(np.allclose(out,out_test,atol=1e-02,rtol=1e-02),
                        msg="This is very fragile and could easily fail due to too much added jitter.")

    
if __name__ == '__main__':
    unittest.main()