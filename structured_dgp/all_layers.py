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
from scipy.linalg import block_diag
from gpflow.params import Parameter, Parameterized, ParamList
from gpflow import transforms
from gpflow import settings 
from gpflow import params_as_tensors, autoflow
from gpflow.features import InducingPoints
import tensorflow as tf
from tensorflow.linalg import LinearOperatorFullMatrix as LOFM
from tensorflow.linalg import LinearOperatorBlockDiag as LOBD
import tensorflow_probability as tfp

"""
The following classes are loosely based on Doubly-Stochastic-DGP V 1.0
( https://github.com/ICL-SML/Doubly-Stochastic-DGP/ 
Copyright 2017 Hugh Salimbeni, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
The main difference is that instead of L layer objects, only one layers object is used
that contains the information of all GPs in all layers.
"""

class Layers(Parameterized):
    def __init__(self, X, Y, all_kernels, **kwargs):
        Parameterized.__init__(self, **kwargs)
        
        #get number of GPs per layer, depends on kernel dimensions        
        self.all_Gps_per_layer = []
        for kernel in all_kernels[1:]:
            dim_out = kernel.input_dim            
            self.all_Gps_per_layer.append(dim_out)
        # final layer
        self.all_Gps_per_layer.append(Y.shape[1])
        self.all_Gps_per_layer = np.array(self.all_Gps_per_layer)
        self.all_Gps_per_layer_tf = tf.constant(self.all_Gps_per_layer,dtype=tf.int32)
        
        self.all_kernels = ParamList(all_kernels)
    
    def tf_kron_2d(self,A, B):
        """
        since there is no tf implementation of the kronecker product, this one is used.
        only for 2d tensors
        """
        M = tf.shape(A)[0]
        N = tf.shape(A)[1]
        P = tf.shape(B)[0]
        R = tf.shape(B)[1]
    
        A_tilde = tf.reshape(A, [M, 1, N, 1])
        B_tilde = tf.reshape(B, [1, P, 1, R])
        C_tilde = A_tilde*B_tilde
        C = tf.reshape(C_tilde, [M*P, N*R])
        return C    
        
    def KL(self):
        return tf.cast(0., dtype=settings.float_type)

class Fully_Coupled_Layers(Layers):
    """
    The layers object for the fully-coupled DGP, stores the variational parameters
    self.mu_M and self.S_M_sqrt and implements the KL divergence term of the ELBO
    in self.KL().
    Furthermore implements the MC sampling through the layers of the DGP
    in self.sample_from_first_layer() and self.sample_from_lth_layer().
    The other functions are only auxiliary for these main tasks.
    Names of variables are chosen to match the variables in the paper as closely as possible.
    """
    def __init__(self, X, Y, Z, all_kernels, all_mfs, all_Zs, mu_M=None, S_M=None,
                 whitened_prior=False, is_full=True, **kwargs):
        Layers.__init__(self, X, Y, all_kernels, **kwargs)
        
        self.all_mean_funcs = ParamList(all_mfs)
        self.all_Zs = all_Zs
        self.all_features = ParamList([InducingPoints(z_l) for z_l in self.all_Zs])
        self.whitened_prior = whitened_prior
        self.needs_build_cholesky = True
        
        M = self.all_Zs[0].shape[0] # nof inducing points
        self.M = M
        T = np.sum(self.all_Gps_per_layer) #nof latent processes
        self.T = T
        L = self.all_Gps_per_layer.shape[0] #nof layers
        self.L = L
        
        #check that mu_M, if given, has shape (M*T,1), otherwise set to 0       
        if mu_M is None:
            self.mu_M = Parameter(np.zeros((M*T,1)))
        else:
            if mu_M.shape != (M*T,1):
                print("Warning: mu_M doesn't have the right shape (M*T,1), set to zeros")
                self.mu_M = Parameter(np.zeros((M*T,1)))
            else:
                self.mu_M = Parameter(mu_M)
                
        #initialisation of variational parameters for fully coupled layers (different for special layers)
        if not is_full:
            return
        
        #check that mu_M, if given, has shape (M*T,M*T), otherwise set to prior 
        transform = transforms.LowerTriangular(M*T, num_matrices=1)
        if S_M is None:
            self.S_M_sqrt = self.S_M_sqrt_para(transform, M)
        else:
            if S_M.shape != (M*T,M*T):
                print("Warning: S_M doesn't have the right shape (M*T,M*T), initialized to prior covariance")
                self.S_M_sqrt = self.S_M_sqrt_para(transform, M)
            else:
                self.S_M_sqrt = Parameter([S_M], transform = transform)
    
    @params_as_tensors    
    def KL(self):
        """ calculate the KL-divergence between the fully coupled variational distribution
        over the inducing outputs q(f_M) = N(mu_M,S_M) and the (block diagonal) posterior
        p(f_M) = \prod_l p(f_M^l), where p(f_M^l) = N(0,K_M^l) (or N(0, I) for the whitened prior)"""
        M, T, L  = self.M, self.T, self.L       
        
        KL = -0.5 * M * T
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.matrix_diag_part(self.S_M_sqrt) ** 2)) #log determinant of S_M
        if self.whitened_prior:
            """KL is much easier to calculate in this case: log det of prior is 0, trace and mean term
            have very simply forms:"""                           
            KL += 0.5 * tf.reduce_sum(tf.square(self.S_M_sqrt)) #trace term
            KL += 0.5 * tf.reduce_sum(self.mu_M**2) #mean term            
        else:
            self.build_cholesky_if_needed()
            for l in range(L):
                Lmm_l = self.all_LMM[l]
                Tl = self.all_Gps_per_layer_tf[l]
                # log determinant of prior (decomposes between layers and tasks)
                KL += 0.5*tf.cast(Tl,settings.float_type)*tf.reduce_sum(tf.log(tf.matrix_diag_part(Lmm_l)**2))
                """precalculations for the trace term:
                the sum of the diagonal-diagonal terms of S_M (MxM matrices) can be cleverly obtained
                using some cholesky tricks (which avoid having to calculate S_M), namely taking the MxM blocks
                in the relevant part of L, i.e., the rows from startind to startind+M*Tl, ignoring the part
                that lies in the upper triangular part (stopind). Then the sum of all the matrix multiplications
                of the MxM blocks (L_i L_i^T) is efficiently performed by transforming into a 'vector'
                of MxM blocks which is multiplied with its transposed."""
                startind = self.give_index(l,0)
                stopind = self.give_index(l+1,0)   
                splits = tf.tile([self.M],[Tl])
                L_vec = tf.concat(tf.split(self.S_M_sqrt[0,startind:(startind+M*Tl),:stopind], splits),axis = 1)
                #trace term
                KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(Lmm_l,L_vec,lower=True)))
                #mean term: triangular solve can solve for batches therefore tile and reshape to avoid loops
                Lmm_l_tiled = tf.tile(Lmm_l[None,:,:],(Tl,1,1))
                mu_M_tiled = tf.reshape(self.mu_M[startind:stopind],(Tl,M,1))
                KL += 0.5 * tf.reduce_sum(tf.square(tf.matrix_triangular_solve(Lmm_l_tiled,
                                                                               mu_M_tiled, lower=True)))
        
        return KL
    
    @params_as_tensors
    def sample_from_first_layer(self, X,S, z=None):
        """
        Implements the equations from Thm. 1 for \widetilde{\mu}_n^1 and \widetilde{S}_n^{11}
        and samples from the resulting multivariate Gaussian using the reparametrization trick.
        Note that \widetilde{\mu}_n^1 and \widetilde{S}_n^{11} are the same for all S samples,
        only the actual sampling from the resulting distribution has to be repaeated.
        
        Inputs:
        data points X, (N,D)
        number of samples S
        (optional) z (S*N,T_1), list of values used for the reparametrization trick for reproducibility
        
        Outputs:
        f1N (S*N, T_1) drawn samples
        mean (S*N,T_1), means used to draw the samples
        var (S*N,T_1,T_1), covariances used to draw the samples
        
        muN1tilde (S*N,T_1), \widetilde{\mu}_n^1 for all n, in the first layer identical with the means
        SN1tilde_chol (S*N,T_1,T_1), Cholesky factors of \widetilde{S}_n^{11}
        KMN_inv (1,M,N), \widetilde{K}^1_{MN} from Eq. (1)
        """
        
        T1 = self.all_Gps_per_layer_tf[0] #nof latent processes in first layer
        M = self.M # nof inducing points
        N = tf.shape(X)[0]
        
        """calculate K1MN_inv"""
        self.build_cholesky_if_needed()        
        KMM = self.all_KMM[0]
        LMM = self.all_LMM[0]        
        KMN = self.all_features[0].Kuf(self.all_kernels[0],X)
        if not self.whitened_prior:
            KMN_inv = tf.cholesky_solve(LMM, KMN) #matmul(KMM_inv,KMN) #M,N
        else:
            KMN_inv = tf.matrix_triangular_solve(LMM, KMN,lower=True) #matmul(LMM_inv,KMN) #M,N
        
        """calculate mu1Ntilde"""
        stopind = self.give_index(1,0)
        KMNtilde = tf.tile(KMN_inv[None,:,:],(T1,1,1))#T1,M,N
        mu_M_tiled = tf.reshape(self.mu_M[0:stopind],[T1,M,1]) #T1,M,1
        muN1tilde = tf.transpose(tf.matmul(KMNtilde,mu_M_tiled,transpose_a=True)[:,:,0],(1,0)) #N,T1
        
        """calculate S1Ntilde"""
        l, startind = 0,0
        SN1tilde = self.SlNtilde(X,l,T1,KMM, KMN_inv,startind=startind,stopind=stopind)
        
        """finally calculate mean and var and then draw from multivariate Gaussian"""
        mean = muN1tilde + self.all_mean_funcs[0](X) #N,T1
        muN1tilde = tf.identity(mean)
        var = SN1tilde #N,T1,T1 or N,T1
        """If we have to draw multiple samples, everything before is exactly the same,
        only here it makes a difference.
        So draw multiple samples and prepare the other quantities for the later layers (tiling)"""
        mean, muN1tilde = tf.tile(mean,[S,1]), tf.tile(muN1tilde,[S,1])        
        KMN_inv = tf.tile(KMN_inv,[1,S])
        f1N, var, SN1tilde_chol = self.draw_samples_first_layer(N,S,T1,mean,var,z)
        
        return f1N, mean, var, muN1tilde, SN1tilde_chol, KMN_inv[None,:,:]
        
    @params_as_tensors
    def sample_from_lth_layer(self, l, fN, all_muNtilde, SNtilde_chol, all_KMN_inv, z=None):
        """
        Implements Eqs. 9,10,11 from the paper, using the reparametrization trick for sampling.
        
        Inputs:
        l, zero-based layer index
        fN (S*N, Tl_bef), f_N^{1:l} samples drawn in previous layers
        all_muNtilde (S*N, Tl_bef), \widetilde{\mu}_n^{1:l} for all samples
        SNtilde_chol (S*N,Tl_bef,Tl_bef), cholesky factor of \widetilde{S}_n^{1:l,1:l} for all samples
        all_KMN_inv (l,M,N), \widetilde{K}^{1:l}_{MN}
        (optional) z (S*N,T_l), list of values used for the reparametrization trick for reproducibility
        
        Outputs:
        f1N (S*N, Tl_bef + T_l) all drawn samples so far
        mulNhat (S*N,T_l), means used to draw the samples of the current layer
        siglNhat (S*N,T_1,T_1), covariances used to draw the samples of the current layer
        
        all_muNtilde (S*N, Tl_bef+ T_l), \widetilde{\mu}_n^{1:l+1} for all samples
        SNtilde_chol (S*N,Tl_bef+T_l,Tl_bef+T_l), cholesky factor of \widetilde{S}_n^{1:l+1,1:l+1} for all samples
        all_KMN_inv (l+1,M,N), \widetilde{K}^{1:l+1}_{MN}
        """
        Tl = self.all_Gps_per_layer_tf[l] #nof latent processes in lth layer
        Tl_bef = tf.reduce_sum(self.all_Gps_per_layer_tf[:l]) #nof latent processes in all layers before
        if l==1:
            Tl_bef2 = 0
        else:
            #nof latent processes in all layers before the last two
            Tl_bef2 = tf.reduce_sum(self.all_Gps_per_layer_tf[:(l-1)]) 
        M = self.M # nof inducing points
        N = tf.shape(fN)[0] #this is actually N*S (but multiple samples are simply treated as additional inputs)
        
        """calculate KlMN_inv"""
        self.build_cholesky_if_needed()
        KMM,LMM = self.all_KMM[l],self.all_LMM[l]
        KMN = self.all_features[l].Kuf(self.all_kernels[l],fN[:,Tl_bef2:Tl_bef]) #M,N
        if not self.whitened_prior:
            KlMN_inv = tf.cholesky_solve(LMM, KMN) #matmul(KMM_inv,KMN) #M,N
        else:
            KlMN_inv = tf.matrix_triangular_solve(LMM, KMN) #matmul(LMM_inv,KMN) #M,N
        
        """calculate mulNtilde"""
        startind, stopind = self.give_index(l,0), self.give_index(l+1,0)
        KlMNtilde = tf.tile(KlMN_inv[None,:,:],(Tl,1,1))#Tl,M,N
        mul_M_tiled = tf.reshape(self.mu_M[startind:stopind],[Tl,M,1]) #Tl,M,1
        mulNtilde = tf.transpose(tf.matmul(KlMNtilde,mul_M_tiled,transpose_a=True)[:,:,0],(1,0)) #N,Tl
        mulNtilde += self.all_mean_funcs[l](fN[:,Tl_bef2:Tl_bef]) #N,Tl #add mean
        
        """calculate SlNtildesubT (SNtilde[l,1:l-1].T), N,Tl_bef,Tl"""
        SlNtildesubT = self.SlNtildesubT(l,Tl,KlMN_inv,all_KMN_inv,startind=startind,stopind=stopind)
        
        """calculate SlNtilde"""
        SlNtilde = self.SlNtilde(fN,l,Tl,KMM, KlMN_inv,startind=startind,stopind=stopind)
        """calculate mean mulNhat and var siglNhat"""
        mulNhat, siglNhat = self.calculate_mean_var(l,SNtilde_chol,SlNtildesubT,mulNtilde,fN,
                                                    all_muNtilde,SlNtilde)
        
        """draw samples from multivariate Gaussian"""
        flN = self.draw_samples_lth_layer(N,Tl,mulNhat,siglNhat,z)
        
        """update all tensors that are necessary for propagating through the layers"""
        all_muNtilde = tf.concat([all_muNtilde,mulNtilde],axis = 1)# N,Tl+Tl_bef
        all_KMN_inv = tf.concat([all_KMN_inv,KlMN_inv[None,:,:]],axis=0) #l,M,N
        fN = tf.concat([fN, flN],axis=1)#N,Tl+Tl_bef        
        
        SNtilde_chol = self.update_cholesky(Tl,SNtilde_chol,SlNtildesubT,SlNtilde)
        
        return fN, mulNhat, siglNhat, all_muNtilde, SNtilde_chol, all_KMN_inv
        
    @params_as_tensors
    def SlNtilde(self,X,l,Tl,KMM,KMN_inv,startind = 0,stopind = 0):
        """
        auxiliary function to calculate \widetilde{S}_n^{l,l} according to the definitions in
        Thm. 1, partly explained in Appx. D.1.
        """
        if l == 0:
            #calculate S_M^11
            COP = tfp.bijectors.CholeskyOuterProduct #allows slightly faster calculation of LL^T        
            SK = COP().forward(x=self.S_M_sqrt[0,0:stopind,0:stopind])#T1*M,T1*M
        else:
            #calculate S_M^ll
            L_vec = self.S_M_sqrt[0,startind:stopind,0:stopind] #Tl*M,(Tl_sum +Tl)*M, corresponds to L[l,1:l]
            SK = tf.matmul(L_vec,L_vec,transpose_b = True)#Tl*M,Tl*M
        #subtract either K_M^l or I (if the prior is whitened) on the diagonals 
        if self.whitened_prior:
            SK -= tf.eye(Tl*tf.shape(KMM)[0], dtype=settings.float_type)
        else:
            SK -= self.tf_kron_2d(tf.eye(Tl, dtype=settings.float_type), KMM) #Tl*M,Tl*M
        splits = tf.tile([self.M],[Tl])
        SK_block = tf.stack(tf.split(tf.stack(tf.split(SK,splits)),splits,axis=2),axis=1) #Tl,Tl,M,M
        """ next, multiply each MxM block from left with KMN_inv^T and from right with KMN_inv
        individually for each n =1,...,N. this is done by using that the diagonal of A @ B @ A.T
        can be obtained by reduce_sum(A@B * A)
        """
        # first step: do (A@B * A) Tl,Tl,M,N (multiply broadcasts and only affects last two dimensions)
        temp = tf.multiply(KMN_inv,tf.tensordot(SK_block,KMN_inv,axes=((3),(0))))
        # second step: reduce_sum() and reshape to N,Tl,Tl
        temp = tf.transpose(tf.transpose(tf.reduce_sum(temp,axis=2),(2,1,0)),(0,2,1))
        # add K1nn to the diagonals of the TlxTl blocks
        KlNN_tiled = tf.tile(self.all_kernels[l].Kdiag(X)[:,None],(1,Tl)) #N,Tl
        SlNtilde = tf.linalg.set_diag(temp,tf.linalg.diag_part(temp)+KlNN_tiled) #N,Tl,Tl
        return SlNtilde
    
    @params_as_tensors
    def draw_samples_first_layer(self,N,S,T1,mean,var,z):
        I_jitter = settings.jitter * tf.eye(T1, dtype=settings.float_type)[None,:,:] #1,T1,T1
        SN1tilde_chol = tf.cholesky(var + I_jitter) #tf addition broadcasts
        if z is None:
            z_s = tf.random_normal([N*S,T1,1], dtype=settings.float_type)
        else:
            z_s = z[:,:,None]
        var, SN1tilde_chol = tf.tile(var,[S,1,1]), tf.tile(SN1tilde_chol,[S,1,1])  
        f1N = mean + tf.linalg.LinearOperatorLowerTriangular(SN1tilde_chol).matmul(z_s)[:,:,0] #S*N,T1
        return f1N, var, SN1tilde_chol  
    
    @params_as_tensors
    def SlNtildesubT(self,l,Tl,KlMN_inv,all_KMN_inv,startind = 0,stopind = 0):
        """
        Calculate the quantity in Eq. (94), or rather its transposed as explained in Appx. D.1.
        """
        stopind1 = self.give_index(1,0) #parameters of the first layer
        #get the Sl1M block by multiplying the corresponding parts of the cholesky,i.e. Ll1L11^T
        Sl1M = tf.matmul(self.S_M_sqrt[0,startind:stopind,0:stopind1],
                         self.S_M_sqrt[0,0:stopind1,0:stopind1],transpose_b=True) #MTl,MT1
        T1 = self.all_Gps_per_layer_tf[0] #nof latent processes in 1st layer
        splitsl = tf.tile([self.M],[Tl])
        splits1 = tf.tile([self.M],[T1])
        Sl1M_block = tf.stack(tf.split(tf.stack(tf.split(Sl1M,splitsl)),splits1,axis=2),axis=1) #Tl,T1,M,M
        #next, multiply each MxM block from left with KlMN_inv^T and from right with K1MN_inv
        #individually for each n =1,...,N. this is done by using that the diagonal of A @ B @ C
        #can be obtained by reduce_sum(A* B@C.T)
        # first step: do (A@B * A) Tl,T1,M,N (multiply broadcasts and only affects last two dimensions)
        SlNtildesub = tf.multiply(KlMN_inv,tf.tensordot(Sl1M_block,all_KMN_inv[0],axes=((3),(0))))
        # second step: reduce_sum() and reshape to N,Tl,T1
        SlNtildesub = tf.transpose(tf.transpose(tf.reduce_sum(SlNtildesub,axis=2),(2,1,0)),(0,2,1))
        # now do the same for the other l2 < l and concat the resulting terms with SlNtildesub
        for l2 in range(1,l):
            startindl2, stopindl2 = self.give_index(l2,0), self.give_index(l2+1,0)
            #get the Sll2M block by multiplying the corresponding parts of the cholesky,
            #i.e. L[l,:l2]L[l2,:l2].T
            Sll2M = tf.matmul(self.S_M_sqrt[0,startind:stopind,0:stopindl2],
                             self.S_M_sqrt[0,startindl2:stopindl2,0:stopindl2],transpose_b=True) #MTl,MTl2
            Tl2 = self.all_Gps_per_layer_tf[l2] #nof latent processes in l2th layer
            splitsl2 = tf.tile([self.M],[Tl2])
            Sll2M_block = tf.stack(tf.split(tf.stack(tf.split(Sll2M,splitsl)),splitsl2,axis=2),axis=1) #Tl,Tl2,M,M
            # first step: do (A* B@C) Tl,Tl2,M,N (multiply broadcasts and only affects last two dimensions)
            temp = tf.multiply(KlMN_inv,tf.tensordot(Sll2M_block,all_KMN_inv[l2],axes=((3),(0))))
            # second step: reduce_sum() and reshape to N,Tl,Tl2
            temp = tf.transpose(tf.transpose(tf.reduce_sum(temp,axis=2),(2,1,0)),(0,2,1))
            SlNtildesub = tf.concat([SlNtildesub,temp], axis = 2)
        #after the loop, SlNtildesub has dimensions N,Tl,Tl_bef
        return tf.matrix_transpose(SlNtildesub) #N,Tl_bef,Tl
    
    @params_as_tensors
    def update_cholesky(self,Tl,SNtilde_chol,SlNtildesubT,SlNtilde):
        """
        Update the cholesky decompositions of \widetilde{S}_n^{1:l-1,1:l-1} for all n
        to obtain those of \widetilde{S}_n^{1:l,1:l} as explained in the second part of Appx. D.1.
        """
        #do the update of the cholesky of SNtilde
        #update consists of concatenating the matrix whose chol.decomp. is given by SNtilde_chol with SlNtilde,
        #SlNtildesub (and its transposed), new chol.decomp. then given by [[SNtilde_chol,0],[L21,L22]]
        L21T = tf.matrix_triangular_solve(SNtilde_chol,SlNtildesubT) # N,Tl_bef,Tl
        
        I_jitter = settings.jitter * tf.eye(Tl, dtype=settings.float_type)[None,:,:] #1,Tl,Tl
        L22 = tf.cholesky(SlNtilde - tf.matmul(L21T,L21T,transpose_a = True) + I_jitter) # N,Tl,Tl
        
        tf_zeros = tf.zeros(tf.shape(L21T),dtype=settings.float_type)
        upper = tf.concat([SNtilde_chol,tf_zeros],axis=2)
        lower = tf.concat([tf.matrix_transpose(L21T),L22],axis=2)
        SNtilde_chol = tf.concat([upper, lower],axis = 1)# N,Tl_bef+Tl,Tl_bef+Tl
        
        return SNtilde_chol
    
    @params_as_tensors
    def calculate_mean_var(self,l,SNtilde_chol,SlNtildesubT,mulNtilde,fN,all_muNtilde,SlNtilde):
        """ This implements Eqs. (10) and (11) after all necessary quantities have been calculated"""
        SlNsubinvT = tf.cholesky_solve(SNtilde_chol,SlNtildesubT) # N,Tl_bef,Tl
        mean_term = (fN - all_muNtilde)[:,:,None]
        mulNhat = mulNtilde + tf.matmul(SlNsubinvT,mean_term,transpose_a=True)[:,:,0] #N,Tl
        siglNhat = SlNtilde - tf.matmul(SlNsubinvT,SlNtildesubT,transpose_a=True) #N,Tl,Tl 
        return mulNhat, siglNhat
    
    @params_as_tensors
    def draw_samples_lth_layer(self,N,Tl,mean,var,z):
        I_jitter = settings.jitter * tf.eye(Tl, dtype=settings.float_type)[None,:,:] #1,Tl,Tl
        siglNhat_chol = tf.cholesky(var + I_jitter)
        #finally draw from multivariate Gaussian
        if z is None:
            z = tf.random_normal([N,Tl,1], dtype=settings.float_type)#N,Tl,1
        else:
            z = z[:,:,None]
        flN = mean + tf.linalg.LinearOperatorLowerTriangular(siglNhat_chol).matmul(z)[:,:,0] #N,Tl
        return flN
    
    def S_M_sqrt_para(self, transform, M):
        """Initialise S_M_sqrt to the prior """
        
        if self.whitened_prior:
            S_M = np.eye(M*np.sum(self.all_Gps_per_layer))[np.newaxis,:]#M*T_tot (total # of GPs in the DGP)
        else:
            L = self.L
            S_M = [block_diag(*[block_diag(*[np.linalg.cholesky(
                            self.all_kernels[l].compute_K_symm(self.all_Zs[l]) + np.eye(M)*settings.jitter) 
                    for _ in range(self.all_Gps_per_layer[l])]) for l in range(L)])]
        return Parameter(S_M, transform = transform)
        
    def give_index(self,l,t=0):
        """
        help function to get the starting index in mu_M and S_M
        for a given layer l and a given latent process (task) t
        """
        M = self.M # nof inducing points        
        ind = M*tf.reduce_sum(self.all_Gps_per_layer_tf[:l])
        return ind
    
    @params_as_tensors
    def build_cholesky_if_needed(self):
        """
        make sure that cholesky of K_{MM} is only calculated once    
        """
        if self.needs_build_cholesky:
            L = self.L #nof layers
            self.all_KMM = [self.all_features[l].Kuu(self.all_kernels[l],
                                                    jitter=settings.jitter) for l in range(L)]
            self.all_LMM = [tf.cholesky(KMM) for KMM in self.all_KMM]
            self.needs_build_cholesky = False
    
class Fast_Stripes_Arrow_Layers(Fully_Coupled_Layers):
    def __init__(self, X, Y, Z, all_kernels, all_mfs, all_Zs, mu_M=None, S_M=None, whitened_prior=False,
                 stripes=True, arrow=True, **kwargs):
        Fully_Coupled_Layers.__init__(self, X, Y, Z, all_kernels, all_mfs, all_Zs, mu_M=mu_M, S_M=S_M,
                                      is_full=False, whitened_prior=whitened_prior, **kwargs)
        """
        The layers object for the efficient STAR DGP, stores the variational
        parameters self.mu_M and the non-zero parts of the lower diagonal S_M_sqrt
        in self.diag_blocks, self.stripe_blocks, and self.arrow_blocks.
        
        self.diag_blocks (T,M,M), stores all the diagonal (lower diagonal) MxM blocks
        self.stripe_blocks (self.num_stripes_blocks,M,M), see below for the calculation
        of self.num_stripes_blocks stores all the MxM blocks in the off-diagonal stripes
        self.arrow_blocks (T-1,M,M) stores all the MxM blocks of the arrowhead
        
        Implements the KL divergence term of the ELBO in self.KL().
        Reuses the MC sampling through the layers, 
        self.sample_from_first_layer() and self.sample_from_lth_layer(),
        from the fully-coupled layers, but efficiently reimplements
        the auxiliary functions for these main tasks.
        """
        self.stripes = stripes
        self.arrow = arrow
        self.M = self.all_Zs[0].shape[0] # nof inducing points
        self.T = np.sum(self.all_Gps_per_layer) #nof latent processes
        self.t = self.all_Gps_per_layer[0] #nof Gps in latent layers
        self.L = self.all_Gps_per_layer.shape[0] #nof layers
        if self.L < 3:
            self.stripes = False
        if self.L < 2:
            self.arrow = False
        if self.whitened_prior:
            diag_blocks = [np.eye(self.M)*1e-5 for l in range(self.L-1) for _ in range(self.t)]
            diag_blocks.append(np.eye(self.M))
        else:
            #initialise diagonal to prior
            LMMs = [np.linalg.cholesky(self.all_kernels[l].compute_K_symm(self.all_Zs[l]) +
                                       np.eye(self.M)*settings.jitter) for l in range(self.L)]
            diag_blocks = [LMMs[l]*1e-5 for l in range(self.L-1) for _ in range(self.t)]
            diag_blocks.append(LMMs[self.L-1])
            diag_blocks = np.array(diag_blocks)
        transform = transforms.LowerTriangular(self.M, num_matrices=self.T)
        self.diag_blocks = Parameter(diag_blocks,transform = transform)
        #stripes and arrow initialised to zero
        """THIS WORKS ONLY FOR SAME NUMBER OF GPS IN LATENT LAYERS """
        if self.stripes:
            stripe_block = np.zeros((self.M,self.M))
            self.num_stripes_blocks = self.t*np.sum([l-1 for l in range(2,self.L)])
            stripe_blocks = np.tile(stripe_block,(self.num_stripes_blocks,1,1))
            self.stripe_blocks = Parameter(stripe_blocks)
        if self.arrow:
            arrow_block = np.zeros((self.M,self.M))
            self.num_arrow_blocks = self.T  - self.all_Gps_per_layer[-1]
            arrow_blocks = np.tile(arrow_block,(self.num_arrow_blocks,1,1))
            self.arrow_blocks = Parameter(arrow_blocks)
    
    """
    In the following the auxiliary functions from the fully coupled layers are reimplemented,
    where the sparsity of S_M_sqrt (mostly for the KL divergence) and also the sparsity of
    \widetilde{S}_n^{1:L,1:L} is exploited. The latter point is explained in more detail in Appx. D.2.
    Otherwise ideas already used for the FC DGP are used again.
    
    Especially depending on whether the arrow or stripes flag are True, certain calculations can
    be considerably simplified. This is also exploited.
    
    Comments like 'currently required' or 'could be turned off for last layer' indicate parts of the
    code where further minor speed-ups could be obtained as discussed
    in the second to last paragraph of Appx. D.2.
    """
    
    @params_as_tensors    
    def KL(self):
        KL = - 0.5 * self.M *self.T #dimension of the multivariate gaussians
        KL -= 0.5 * tf.reduce_sum(tf.log(tf.square(tf.matrix_diag_part(self.diag_blocks)))) #log det of S_M
        if self.whitened_prior:
            """KL is much easier to calculate in this case: log det of prior is 0, trace and mean term
            have very simply forms:"""
            #sum only non zero elements for trace term
            KL += 0.5 * tf.reduce_sum(tf.square(self.diag_blocks))
            if self.arrow:
                KL += 0.5 * tf.reduce_sum(tf.square(self.arrow_blocks))
            if self.stripes:
                KL += 0.5 * tf.reduce_sum(tf.square(self.stripe_blocks))
            KL += 0.5 * tf.reduce_sum(self.mu_M**2) #mean term
        else:
            self.build_cholesky_if_needed()
            for l in range(self.L):
                Lmm_l = self.all_LMM[l]
                Tl = self.all_Gps_per_layer_tf[l]
                # log determinant of prior (decomposes between layers and tasks)
                KL += 0.5*tf.cast(Tl,settings.float_type)*tf.reduce_sum(tf.log(tf.matrix_diag_part(Lmm_l)**2))                
                #mean term: triangular solve can solve for batches therefore tile and reshape to avoid loops
                startind = self.give_index(l,0)
                stopind = self.give_index(l+1,0)
                Lmm_l_tiled = tf.tile(Lmm_l[None,:,:],(Tl,1,1))
                mu_M_tiled = tf.reshape(self.mu_M[startind:stopind],(Tl,self.M,1))
                A = tf.matrix_triangular_solve(Lmm_l_tiled, mu_M_tiled, lower=True)
                KL += 0.5 * tf.reduce_sum(tf.square(A))
                """ see trace term of the FC layers for the idea of the calculations below, additionally
                exploit that many terms vanish"""            
                if l == 0: #first layer, only diagonal blocks (-> as in mean-field)
                    temp = tf.matrix_triangular_solve(Lmm_l_tiled, self.diag_blocks[:Tl], lower=True)
                    KL += 0.5 * tf.reduce_sum(tf.square(temp))                    
                elif l == self.L-1: #last layer
                    if self.arrow: #assemble all non-zero blocks to a block vector L_vec
                        arrows = [self.arrow_blocks[i] for i in range(self.num_arrow_blocks)]
                        arrow_block = tf.concat(arrows,axis=1)
                        L_vec = tf.concat([arrow_block,self.diag_blocks[-1]],axis=1)
                        KL += 0.5*tf.reduce_sum(tf.square(tf.matrix_triangular_solve(Lmm_l,L_vec,lower=True)))
                    else: #only diagonal blocks (-> as in mean-field)
                        temp = tf.matrix_triangular_solve(Lmm_l, self.diag_blocks[-1], lower=True)
                        KL += 0.5 * tf.reduce_sum(tf.square(temp))
                else: #all the other layers in between
                    if self.stripes: #assemble all non-zero blocks to a block vector L_vec
                        diags = tf.concat([self.diag_blocks[i] for i in range(self.t*l,self.t*(l+1))],axis=1)
                        startind = int(self.t*l*(l-1)/2) #little Gauss (\sum_(k=1)^(l-1) k)
                        stopind = int(self.t*l*(l+1)/2) #little Gauss (\sum_(k=1)^l k)
                        stripes = tf.concat([self.stripe_blocks[i] for i in range(startind,stopind)],axis=1)
                        L_vec = tf.concat([diags,stripes],axis=1)
                        KL += 0.5*tf.reduce_sum(tf.square(tf.matrix_triangular_solve(Lmm_l,L_vec,lower=True)))
                    else: #as is the first layer
                        diags = self.diag_blocks[self.t*l:self.t*(l+1)]
                        temp=tf.matrix_triangular_solve(Lmm_l_tiled,diags,lower=True)
                        KL += 0.5 * tf.reduce_sum(tf.square(temp))
        return KL
        
    @params_as_tensors
    def SlNtilde(self,X,l,Tl,KMM,KMN_inv,startind = 0,stopind = 0):
        """For all of the chosen  approximations of S_M, S_M^ll is block diagonal.
        Depending on the chosen structure, the (Tl,M,M) tensor S_M^ll has to be obtained in different ways,
        but we can always get it by a batch matmul mat.mat^T where mat has size(Tl,M,x*M), where x
        depends on the structure. In the first part mat is found where sparsity is exploited,
        where Eqs. 107 and 108 are used.
        In the second part the S_N^l are obtained with the same idea as for full S_M.
        Since only the diagonal elements are non-zero, some simplifications can be made (see also Salimbeni)"""
        if l == 0:#first layer is always diagonal
            mat = self.diag_blocks[:Tl] #T1,M,M
        elif l == self.L-1: #last layer (depends on arrow), according to Eq. 108
            if self.arrow:
                arrow_block = tf.concat([self.arrow_blocks[i] for i in range(self.num_arrow_blocks)],axis=1)#M,(T-1)*M
                mat = tf.concat([arrow_block[None,:,:],self.diag_blocks[-1:]],axis=2)#1,M,T*M
                
            else:
                mat = self.diag_blocks[-1:] #1,M,M
        else: #intermediate layers depend on stripes, according to Eq. 107
            diags = self.diag_blocks[l*self.t:(l+1)*self.t] #Tl,M,M
            if self.stripes:
                startind = int(self.t*l*(l-1)/2) #little Gauss (\sum_(k=1)^(l-1) k)
                stopind = int(self.t*l*(l+1)/2) #little Gauss (\sum_(k=1)^l k)
                stripes = tf.concat(tf.split(self.stripe_blocks[startind:stopind],l,axis=0),axis=2)
                mat = tf.concat([diags,stripes],axis=2) #Tl,M,(l+1)*M
            else:
                mat = diags
                
        S_Mll = tf.matmul(mat, mat, transpose_b=True) #Tl,M,M
        
        if self.whitened_prior:
            SK = S_Mll - tf.eye(self.M, dtype=settings.float_type)[None,:,:]
        else:
            SK = S_Mll - KMM[None,:,:]
        KMN_inv_tiled = tf.tile(KMN_inv[None,:,:],[Tl,1,1]) #Tl,M,N
        temp = tf.multiply(KMN_inv_tiled,tf.matmul(SK,KMN_inv_tiled))#Tl,M,N
        temp = tf.reduce_sum(temp,axis=1)#Tl,N
        KlNN_tiled = self.all_kernels[l].Kdiag(X)[None,:] #Tl,N
        SlNtilde = tf.transpose(temp + KlNN_tiled) #N,Tl
        """currently required"""
        if not self.stripes:#mean field, no cholesky of SNtilde needed
            return SlNtilde
        if l > 0:
            SlNtilde = tf.matrix_diag(SlNtilde)

        return SlNtilde
    
    @params_as_tensors
    def draw_samples_first_layer(self,N,S,T1,mean,var,z):
        """first layer same as mean field """
        std = (var + settings.jitter) ** 0.5
        z_s = z #N*S,T1
        if z_s is None:
            z_s = tf.random_normal([N*S,T1], dtype=settings.float_type)
        var, std = tf.tile(var,[S,1]), tf.tile(std,[S,1])
        f1N = mean + z_s * std #N*S,T1
        """currently required"""
        if not (self.stripes or self.arrow):#mean field, no cholesky of SNtilde needed
            SN1tilde_chol = None
        elif not self.stripes and self.arrow:
            SN1tilde_chol = std #only diagonals needed for last layer
        else:
            SN1tilde_chol = tf.matrix_diag(std)
            
        return f1N, var, SN1tilde_chol
    
    @params_as_tensors
    def update_cholesky(self,Tl,SNtilde_chol,SlNtildesubT,SlNtilde):
        """ efficient implementations for special cases,could be turned off for last layer"""
        if not (self.stripes or self.arrow):#mean field, no cholesky of SNtilde needed
            return None
        elif not self.stripes and self.arrow:
            return tf.concat([SNtilde_chol,(SlNtilde+settings.jitter)**0.5],axis=1) #N,Tl_bef+Tl
        
        """general case, same as for fully coupled layers"""
        #do the update of the cholesky of SNtilde
        #update consists of concatenating the matrix whose chol.decomp. is given by SNtilde_chol with SlNtilde,
        #SlNtildesub (and its transposed), new chol.decomp. then given by [[SNtilde_chol,0],[L21,L22]]
        L21T = tf.matrix_triangular_solve(SNtilde_chol,SlNtildesubT) # N,Tl_bef,Tl
        
        I_jitter = settings.jitter * tf.eye(Tl, dtype=settings.float_type)[None,:,:] #1,Tl,Tl
        L22 = tf.cholesky(SlNtilde - tf.matmul(L21T,L21T,transpose_a = True) + I_jitter) # N,Tl,Tl
        
        tf_zeros = tf.zeros(tf.shape(L21T),dtype=settings.float_type)
        upper = tf.concat([SNtilde_chol,tf_zeros],axis=2)
        lower = tf.concat([tf.matrix_transpose(L21T),L22],axis=2)
        SNtilde_chol = tf.concat([upper, lower],axis = 1)# N,Tl_bef+Tl,Tl_bef+Tl
        
        return SNtilde_chol
    
    @params_as_tensors
    def calculate_mean_var(self,l,SNtilde_chol,SlNtildesubT,mulNtilde,fN,all_muNtilde,SlNtilde):
        """ efficient implementations for special cases"""
        if not self.stripes:
            if not self.arrow:
                return mulNtilde, SlNtilde
            else:
                if l != self.L-1:
                    return mulNtilde, SlNtilde
                else:
                    common_term = tf.multiply(tf.square(1/SNtilde_chol),SlNtildesubT) #N,(L-1)*t
                    mulNhat_add = tf.multiply(common_term,fN - all_muNtilde) #N,(L-1)*t
                    mulNhat_add = tf.reduce_sum(mulNhat_add,axis=1)[:,None] #N,1
                    siglNhat_minus = tf.multiply(common_term,SlNtildesubT) #N,(L-1)*t
                    siglNhat_minus = tf.reduce_sum(siglNhat_minus,axis=1)[:,None] #N,1
                    return mulNtilde + mulNhat_add, SlNtilde - siglNhat_minus
        
        """general case, same as for fully-coupled layers"""    
        SlNsubinvT = tf.cholesky_solve(SNtilde_chol,SlNtildesubT) # N,Tl_bef,Tl
        mean_term = (fN - all_muNtilde)[:,:,None]
        mulNhat = mulNtilde + tf.matmul(SlNsubinvT,mean_term,transpose_a=True)[:,:,0] #N,Tl
        siglNhat = SlNtilde - tf.matmul(SlNsubinvT,SlNtildesubT,transpose_a=True) #N,Tl,Tl 
        return mulNhat, siglNhat
    
    @params_as_tensors
    def draw_samples_lth_layer(self,N,Tl,mean,var,z):
        """ efficient implementations for special cases"""
        if not self.stripes:#mean field, no cholesky of SNtilde needed
            std = (var + settings.jitter) ** 0.5
            z_s = z #N,Tl
            if z_s is None:
                z_s = tf.random_normal([N,Tl], dtype=settings.float_type)
            return mean + z_s * std #N, Tl
        
        """general case, same as for fully-coupled layers"""
        I_jitter = settings.jitter * tf.eye(Tl, dtype=settings.float_type)[None,:,:] #1,Tl,Tl
        siglNhat_chol = tf.cholesky(var + I_jitter)
        #finally draw from multivariate Gaussian
        if z is None:
            z = tf.random_normal([N,Tl,1], dtype=settings.float_type)#N,Tl,1
        else:
            z = z[:,:,None]            
        flN = mean + tf.linalg.LinearOperatorLowerTriangular(siglNhat_chol).matmul(z)[:,:,0] #N,Tl
        return flN
    
    @params_as_tensors
    def SlNtildesubT(self,l,Tl,KlMN_inv,all_KMN_inv,startind = 0,stopind = 0):
        """currently required"""
        if not (self.stripes or self.arrow):#mean field, no off-diagonal terms present
            return None
        N = tf.shape(KlMN_inv)[1]
        if l == self.L - 1 : #last layer
            if self.arrow:
                SlNtildesubT = self.SlNtildesub_last_layer(l,KlMN_inv,all_KMN_inv)#N,(L-1)*t
                """currently required"""
                if self.stripes:
                    SlNtildesubT = SlNtildesubT[:,:,None]                    
            else:
                SlNtildesubT = None
                """currently required"""
                SlNtildesubT = tf.zeros([N,(self.L-1)*self.t,1],dtype=settings.float_type)
        else: #all other layers
            if self.stripes:
                SlNtildesubT = self.SlNtildesub_intermediate_layer(l,KlMN_inv,all_KMN_inv)#N,(l-1)*t
                """currently required"""
                SlNtildesubT = tf.reshape(SlNtildesubT,[-1,l,self.t])
                SlNtildesubT = tf.matrix_diag(SlNtildesubT) #N,l-1,t,t
                SlNtildesubT = tf.concat(tf.split(SlNtildesubT,l,axis=1),axis=3)[:,0,:,:]
                SlNtildesubT = tf.matrix_transpose(SlNtildesubT)#N,(l-1)*t,t
            else:
                SlNtildesubT = None                
        return SlNtildesubT
    
    @params_as_tensors
    def SlNtildesub_last_layer(self,l,KlMN_inv,all_KMN_inv):
        """
        special implementation, calculating the terms of \widetilde{S}_n^{L,1:L}.
        The basic idea is to use the equivalent Eq. 95 and exploit that the terms
        used for the last layer can be efficiently obtained using Eqs. 110.
        """
        arrows = self.arrow_blocks[:self.t] #t,M,M
        diags = self.diag_blocks[:self.t] #t,M,M
        SML1 = tf.matmul(arrows,diags,transpose_b=True) #t,M,M
        SMLK_sub = tf.tensordot(SML1,all_KMN_inv[0],axes=[[2],[0]]) #t,M,N
        #(This is actually SML1.KMN1_inv, the other matrices are later concatenated with it)
        for l2 in range(1,l):
            #one term coming from the diagonal (first term in Eq. 110)
            diag_arrows = self.arrow_blocks[l2*self.t:self.t*(l2+1)] #t,M,M
            diags = self.diag_blocks[l2*self.t:self.t*(l2+1)] #t,M,M
            SMLl2 = tf.matmul(diag_arrows,diags,transpose_b=True) #t,M,M
            #other terms coming from the stripes
            if self.stripes:
                #only certain blocks contribute to the results, those have the indices inds
                #and can be gathered using gather_nd, this is the second part of Eq. 110
                inds = [ind for t in range(self.t) for ind in range(self.t*l2) if ind % self.t == t]
                inds = tf.constant(inds,dtype=tf.int32)[:,None]
                arrows_permuted = tf.gather_nd(self.arrow_blocks[:self.t*l2],inds) #l2*t,M,M
                arrows = tf.reshape(arrows_permuted,[l2,self.t,self.M,self.M]) #l2,t,M,M
                start,stop=int(self.t*l2*(l2-1)/2),int(self.t*l2*(l2+1)/2) #Gauss (\sum_(k=1)^l k)
                stripes_permuted = tf.gather_nd(self.stripe_blocks[start:stop],inds) #l2*t,M,M
                stripes = tf.reshape(stripes_permuted,[l2,self.t,self.M,-1]) #l2,t,M,M
                SMLl2 += tf.reduce_sum(tf.matmul(arrows,stripes,transpose_b=True),axis=0) #t,M,M
            SMLl2K = tf.tensordot(SMLl2,all_KMN_inv[l2],axes=[[2],[0]]) #t,M,N
            SMLK_sub = tf.concat([SMLK_sub,SMLl2K],axis=0) #t*l2,M,N
        #at the end of the loop SML_sub has dimension #t*(L-1),M,N
        #multiply elementwise with KLMN_inv and reduce the dimension which has M elements
        SlNtildesub = tf.transpose(tf.reduce_sum(tf.multiply(KlMN_inv,SMLK_sub),axis=1)) #N,t*(L-1)
        return SlNtildesub
    
    @params_as_tensors
    def SlNtildesub_intermediate_layer(self,l,KlMN_inv,all_KMN_inv):
        """
        special implementation, calculating the terms of \widetilde{S}_n^{l,1:l}.
        The basic idea is to use Eq. 95 and exploit the sparsity using Eq. 109.
        """
        startl = int(self.t*l*(l-1)/2) #little Gauss (\sum_(k=1)^(l-1) k)
        stripes1 = self.stripe_blocks[startl:startl+self.t] #t,M,M
        diags1 = self.diag_blocks[:self.t] #t,M,M
        S_Ml1 = tf.matmul(stripes1, diags1, transpose_b=True) #t,M,M, first part in Eq. 109
        SMlK_sub = tf.tensordot(S_Ml1,all_KMN_inv[0],axes=[[2],[0]]) #t,M,N
        for l2 in range(1,l):
            #only certain blocks contribute to the results, those have the indices inds
            #and can be gathered using gather_nd, this is the second part of Eq. 109
            inds = [ind for t in range(self.t) for ind in range(self.t*(l2+1)) if ind % self.t == t]
            inds = tf.constant(inds,dtype=tf.int32)[:,None]
            
            stopl = startl + self.t * (l2+1)
            stripesl_permuted = tf.gather_nd(self.stripe_blocks[startl:stopl],inds) #l2*t,M,M
            stripesl = tf.reshape(stripesl_permuted,[self.t,l2+1,self.M,self.M]) #t,l2,M,M
            
            startl2, stopl2 = int(self.t*l2*(l2-1)/2), int(self.t*l2*(l2+1)/2)
            stripesl2 = self.stripe_blocks[startl2:stopl2] #t*(l2-1),M,M
            diagl2 = self.diag_blocks[self.t*l2:self.t*(l2+1)] #t,M,M
            stripes_and_diag = tf.concat([stripesl2,diagl2],axis=0) #t*l2,M,M                    
            stripes_and_diag_permuted = tf.gather_nd(stripes_and_diag,inds) #l2*t,M,M                    
            stripes_and_diag = tf.reshape(stripes_and_diag_permuted,[self.t,l2+1,self.M,-1]) #t,l2,M,M
            
            S_Mll2 = tf.matmul(stripesl,stripes_and_diag,transpose_b = True) #t,l2,M,M
            S_Mll2 = tf.reduce_sum(S_Mll2,axis=1)#t,M,M
            SMll2K = tf.tensordot(S_Mll2,all_KMN_inv[l2],axes=[[2],[0]]) #t,M,N
            SMlK_sub = tf.concat([SMlK_sub,SMll2K],axis=0) #t*l2,M,N
        #at the end of the loop SML_sub has dimension #t*(l-1),M,N
        #multiply elementwise with KLMN_inv and reduce the dimension which has M elements
        SlNtildesub = tf.transpose(tf.reduce_sum(tf.multiply(KlMN_inv,SMlK_sub),axis=1)) #N,t*(l-1)
        return SlNtildesub

    @property
    @params_as_tensors
    def S_M_sqrt(self):
        """
        Auxiliary function for building S_M_sqrt from the non-zero blocks.
        Not actually used in the calculations only useful for visualizations.
        """
        if self.stripes:
            #build the matrix layer by layer (except for the last)
            res = LOBD([LOFM(self.diag_blocks[i]) for i in range(self.t)]).to_dense()
            for l in range(1,self.L-1):
                #first concatenate zeros in the part above the diagonal
                zero_block = tf.zeros((self.t*self.M*l,self.t*self.M),dtype=settings.float_type)
                res = tf.concat([res,zero_block],axis=1)
                ind = int(l*(l-1)/2) #little Gauss (\sum_(k=1)^l k)
                blocks = [LOFM(self.stripe_blocks[i]) for i in range(self.t*ind,self.t*(ind+1))]
                stripes_block = LOBD(blocks).to_dense()
                for l2 in range(1,l):
                    ind2 = ind + l2
                    blocks = [LOFM(self.stripe_blocks[i]) for i in range(self.t*ind2,self.t*(ind2+1))]
                    concat_block = LOBD(blocks).to_dense()
                    stripes_block = tf.concat([stripes_block,concat_block],axis=1)
                diag_block = LOBD([LOFM(self.diag_blocks[i]) for i in range(l*self.t,(l+1)*self.t)]).to_dense()
                new_layer_block = tf.concat([stripes_block,diag_block],axis = 1)
                res = tf.concat([res, new_layer_block],axis=0)
            zero_block = tf.zeros((self.t*self.M*(self.L-1),self.M),dtype=settings.float_type)
            res = tf.concat([res,zero_block],axis=1)
            if self.arrow:
                #build last layer from arrow blocks and the last diagonal block
                arrow_block = tf.concat([self.arrow_blocks[i] for i in range(self.num_arrow_blocks)],axis=1)
                last_layer_block = tf.concat([arrow_block,self.diag_blocks[-1]],axis=1)
            else:
                #last layer in this case only has the diagonal element                
                last_layer_block = tf.concat([tf.transpose(zero_block),self.diag_blocks[-1]],axis=1)
            res = tf.concat([res, last_layer_block],axis=0)
        else:
            if self.arrow:
                #everything except the last layer is only diagonal
                res = LOBD([LOFM(self.diag_blocks[i]) for i in range(self.num_arrow_blocks)]).to_dense()
                #add zeros on upper diagonal
                zero_block = tf.zeros((self.t*self.M*(self.L-1),self.M),dtype=settings.float_type)
                res = tf.concat([res,zero_block],axis=1)
                #build last layer from arrow blocks and the last diagonal block
                arrow_block = tf.concat([self.arrow_blocks[i] for i in range(self.num_arrow_blocks)],axis=1)
                last_layer_block = tf.concat([arrow_block,self.diag_blocks[-1]],axis=1)
                res = tf.concat([res, last_layer_block],axis=0)
            else:
                #this is the mean-field DGP
                res = LOBD([LOFM(self.diag_blocks[i]) for i in range(self.T)]).to_dense()
        return tf.expand_dims(res,0)
            
    @params_as_tensors
    @autoflow()
    def S_M_sqrt_value(self):
        """
        Get the value of the tensor.
        """
        return self.S_M_sqrt
    
class Stripes_Arrow_Layers(Fully_Coupled_Layers):
    def __init__(self, X, Y, Z, all_kernels, all_mfs, all_Zs, mu_M=None, S_M=None, whitened_prior=False,
                 stripes=True, arrow=True, **kwargs):
        Fully_Coupled_Layers.__init__(self, X, Y, Z, all_kernels, all_mfs, all_Zs, mu_M=mu_M, S_M=S_M,
                                      is_full=False, whitened_prior=whitened_prior, **kwargs)
        """
        The layers object for the naive STAR DGP, stores the variational
        parameters self.mu_M and the non-zero parts of the lower diagonal S_M_sqrt
        in self.diag_blocks, self.stripe_blocks, and self.arrow_blocks.
        
        Only implements S_M_sqrt, i.e. how to build the full covariance matrix
        (or rather its Cholesky factor) from the non-zero elements.
        This is sufficient to do all the calculations for the fully_coupled_layers.
        
        While this is clearly not efficient, it shows exemplarily how all forms of covariance
        matrices can be tested with a few lines of code.
        """
        self.stripes = stripes
        self.arrow = arrow
        self.M = self.all_Zs[0].shape[0] # nof inducing points
        self.T = np.sum(self.all_Gps_per_layer) #nof latent processes
        self.t = self.all_Gps_per_layer[0] #nof Gps in latent layers
        self.L = self.all_Gps_per_layer.shape[0] #nof layers
        if self.L < 3:
            self.stripes = False
        if self.L < 2:
            self.arrow = False
        if self.whitened_prior:
            diag_blocks = [np.eye(self.M)*1e-5 for l in range(self.L-1) for _ in range(self.t)]
            diag_blocks.append(np.eye(self.M))
        else:
            #initialise diagonal to prior
            LMMs = [np.linalg.cholesky(self.all_kernels[l].compute_K_symm(self.all_Zs[l]) +
                                       np.eye(self.M)*settings.jitter) for l in range(self.L)]
            diag_blocks = [LMMs[l]*1e-5 for l in range(self.L-1) for _ in range(self.t)]
            diag_blocks.append(LMMs[self.L-1])
            diag_blocks = np.array(diag_blocks)
        transform = transforms.LowerTriangular(self.M, num_matrices=self.T)
        self.diag_blocks = Parameter(diag_blocks,transform = transform)
        #stripes initialised to zero
        """ONLY FOR SAME NUMBER OF GPS IN LATENT LAYERS """
        if self.stripes:
            stripe_block = np.zeros((self.M,self.M))
            self.num_stripes_blocks = self.t*np.sum([l-1 for l in range(2,self.L)])
            stripe_blocks = np.tile(stripe_block,(self.num_stripes_blocks,1,1))
            self.stripe_blocks = Parameter(stripe_blocks)
        if self.arrow:
            arrow_block = np.zeros((self.M,self.M))
            self.num_arrow_blocks = self.T  - self.all_Gps_per_layer[-1]
            arrow_blocks = np.tile(arrow_block,(self.num_arrow_blocks,1,1))
            self.arrow_blocks = Parameter(arrow_blocks)
    
    @property
    @params_as_tensors
    def S_M_sqrt(self):
        if self.stripes:
            #build the matrix layer by layer (except for the last)
            res = LOBD([LOFM(self.diag_blocks[i]) for i in range(self.t)]).to_dense()
            for l in range(1,self.L-1):
                #first concatenate zeros in the part above the diagonal
                zero_block = tf.zeros((self.t*self.M*l,self.t*self.M),dtype=settings.float_type)
                res = tf.concat([res,zero_block],axis=1)
                ind = int(l*(l-1)/2) #little Gauss (\sum_(k=1)^l k)
                blocks = [LOFM(self.stripe_blocks[i]) for i in range(self.t*ind,self.t*(ind+1))]
                stripes_block = LOBD(blocks).to_dense()
                for l2 in range(1,l):
                    ind2 = ind + l2
                    blocks = [LOFM(self.stripe_blocks[i]) for i in range(self.t*ind2,self.t*(ind2+1))]
                    concat_block = LOBD(blocks).to_dense()
                    stripes_block = tf.concat([stripes_block,concat_block],axis=1)
                diag_block = LOBD([LOFM(self.diag_blocks[i]) for i in range(l*self.t,(l+1)*self.t)]).to_dense()
                new_layer_block = tf.concat([stripes_block,diag_block],axis = 1)
                res = tf.concat([res, new_layer_block],axis=0)
            zero_block = tf.zeros((self.t*self.M*(self.L-1),self.M),dtype=settings.float_type)
            res = tf.concat([res,zero_block],axis=1)
            if self.arrow:
                #build last layer from arrow blocks and the last diagonal block
                arrow_block = tf.concat([self.arrow_blocks[i] for i in range(self.num_arrow_blocks)],axis=1)
                last_layer_block = tf.concat([arrow_block,self.diag_blocks[-1]],axis=1)
            else:
                #last layer in this case only has the diagonal element                
                last_layer_block = tf.concat([tf.transpose(zero_block),self.diag_blocks[-1]],axis=1)
            res = tf.concat([res, last_layer_block],axis=0)
        else:
            if self.arrow:
                #everything except the last layer is only diagonal
                res = LOBD([LOFM(self.diag_blocks[i]) for i in range(self.num_arrow_blocks)]).to_dense()
                #add zeros on upper diagonal
                zero_block = tf.zeros((self.t*self.M*(self.L-1),self.M),dtype=settings.float_type)
                res = tf.concat([res,zero_block],axis=1)
                #build last layer from arrow blocks and the last diagonal block
                arrow_block = tf.concat([self.arrow_blocks[i] for i in range(self.num_arrow_blocks)],axis=1)
                last_layer_block = tf.concat([arrow_block,self.diag_blocks[-1]],axis=1)
                res = tf.concat([res, last_layer_block],axis=0)
            else:
                #this is the mean-field DGP
                res = LOBD([LOFM(self.diag_blocks[i]) for i in range(self.T)]).to_dense()
        return tf.expand_dims(res,0)
            
    @params_as_tensors
    @autoflow()
    def S_M_sqrt_value(self):
        return self.S_M_sqrt
