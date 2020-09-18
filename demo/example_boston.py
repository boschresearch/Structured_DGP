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

from sklearn.datasets import load_boston

from scipy.cluster.vq import kmeans2
from scipy.linalg import block_diag
from scipy.stats import norm
from scipy.special import logsumexp

from gpflow.kernels import RBF
from gpflow.likelihoods import Gaussian
from gpflow.training import AdamOptimizer

import tensorflow as tf

import matplotlib.pyplot as plt

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from structured_dgp.full_dgp import Full_DGP, Fast_Approx_Full_DGP

def prepare_boston(seed = None, M = 30):
    """
    Load and prepare the boston uci data set: split it in a train and test set,
    normalize the data according to the training data,
    and finally initialize the inducing points.
    Inputs are a seed for a reproducible split and the number of inducing points M.
    """
    #random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
    
    #load the boston data set
    X, y = load_boston(return_X_y=True)
    
    #random 90/10 train/test split
    rd = np.random.permutation(X.shape[0])
    X, y = X[rd], y[rd]
    start = int(0.1*X.shape[0])

    X_test, y_test = X[:start], y[:start]
    X, y = X[start:], y[start:]
    
    #normalize the data set (according to the training data)
    
    X_std, y_std = np.copy(X), np.copy(y)

    X_test -= X.mean(axis=0)
    X_test /= X.std(axis=0)

    y_test -= y.mean()
    y_test /= y.std()

    X_std -= X_std.mean(axis=0)
    X_std /= X_std.std(axis=0)

    y_std -= y_std.mean()
    y_std /= y_std.std()

    y_std, y_test = y_std[:,np.newaxis], y_test[:,np.newaxis]
    
    #initialize the M inducing points with kmeans
    Z, _ = kmeans2(X_std, M, iter=30, minit = '++')
    
    return X_std, y_std, X_test, y_test, y.std(), Z

def prepare_models(X, y, Z, M = 30):
    """
    Initialize three models, FC, STAR, and MF DGP with the same architecture,
    three layers with 2 GPs in the first and second layer, respectively,
    and one GP in the final layer.
    The variational parameters are initialized as described in e.g.
    https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/demos/demo_regression_UCI.ipynb
    making the training more effective in the beginning.
    """
    #prepare the kernels (3 layers, 2 GPs in both hidden layers)    
    dim_X = X.shape[1]
    Ks = [RBF(dim_X, ARD=True, lengthscales=1)]
    Ks += [RBF(2, ARD=True, lengthscales=1),RBF(2, ARD=True, lengthscales=1)]
    
    #fully-coupled
    fc_dgp = Full_DGP(X, y, Z.copy(), Ks.copy(), Gaussian(0.01))
    #start the inner layers almost deterministically 
    SM_prior = fc_dgp.layers.S_M_sqrt.value
    SM_det = block_diag(SM_prior[0,:-M,:-M]* 1e-5,SM_prior[0,-M:,-M:])
    fc_dgp.layers.S_M_sqrt = [SM_det]
    
    #STAR (inner layers are by default initialized almost deterministically)
    star_dgp = Fast_Approx_Full_DGP(X, y, Z.copy(), Ks.copy(), Gaussian(0.01),
                                    stripes = True, arrow = True)
    
    #MF (inner layers are by default initialized almost deterministically)
    mf_dgp = Fast_Approx_Full_DGP(X, y, Z.copy(), Ks.copy(), Gaussian(0.01),
                                    stripes = False, arrow = False)
    
    return fc_dgp, star_dgp, mf_dgp

def sm_plots(models,save=False):
    """
    Generate plots of the covariance matrices of the variational distributions
    of the three models. For better visualization we plot the natural
    logarithms of the absolute values of the individual matrix elements.
    """
    #get the cholesky factors and then calculate the covariance matrices
    sm_fc = models[0].layers.S_M_sqrt.value[0,:,:]
    sm_fc = sm_fc @ sm_fc.T
    
    #note the slight difference in the syntax
    sm_star = models[1].layers.S_M_sqrt_value()[0,:,:]
    sm_star = sm_star @ sm_star.T
    
    sm_mf = models[2].layers.S_M_sqrt_value()[0,:,:]
    sm_mf = sm_mf @ sm_mf.T    

    fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    im0 = axs[0].imshow(np.log(np.abs(sm_fc)), cmap='viridis')
    axs[0].set_title('FC DGP')
    fig.colorbar(im0,ax=axs[0])

    im1 = axs[1].imshow(np.log(np.abs(sm_star)), cmap='viridis')
    axs[1].set_title('STAR DGP')
    fig.colorbar(im1,ax=axs[1])

    im2 = axs[2].imshow(np.log(np.abs(sm_mf)), cmap='viridis')
    axs[2].set_title('MF DGP')
    fig.colorbar(im2,ax=axs[2])
    if save:
        plt.savefig('sm_plots.png',quality=95,dpi=400,bbox_inches = "tight")
        plt.close()
    else:
        plt.show()
    
    return

def calculate_tll(model, X_test, y_test, y_sigma, num_samples=10):
    """
    Evaluate the trained models on the test data set using the (average) test
    log likelihood as a metric. For a Gaussian likelihood it can be calculated
    in a numerically stable way as
    \frac{1}{N} \sum_{n=1}^N \log \sum{s=1}^S \exp \log \mathcal{N} (y_n|f_{n,s}^L,(\sigma_{n,s}^L)^2),
    where we sample every test point S (=num_samples) times through the DGP.
    """
        
    mean, var = model.predict_y(X_test, num_samples)

    # test log-likelihood (with rescaled output values), calculated numerically more stable
    tll_SN = np.sum(norm.logpdf(y_sigma*y_test, loc = y_sigma*mean, scale = y_sigma*var**0.5), 2)
    tll = logsumexp(tll_SN, 0, b=1/float(num_samples))
    
    return np.average(tll)

if __name__ == "__main__":
    """
    Very basic demo, applying different DGP models to the uci boston data set.
    After preparing the data and the models (FC, STAR, MF), all models are trained.
    The inferred covariance matrices of the variational distributions are plotted
    and the average test log likelihoods for the models are estimated.
    For simplicity the (in this case suboptimal) Adam optimizer is used.
    For more information on how natural gradients can be used,
    and how potential problems can be circumvented, use the following links.
    https://github.com/ICL-SML/Doubly-Stochastic-DGP/blob/master/demos/demo_regression_UCI.ipynb
    https://github.com/GPflow/GPflow/issues/820
    """
    #prepare the data and the inducing points
    M = 30 #number of inducing points
    X, y, X_test, y_test, y_sigma, Z = prepare_boston(seed = 1337, M = M)
    
    #prepare the models
    models = prepare_models(X, y, Z, M = M)
    
    #train the models
    lr = 0.015
    for model in models:
        opt = AdamOptimizer(lr)
        opt.minimize(model,maxiter=4000)
    
    #plot the covariance matrices
    #if save is True, a .png is saved in the same directory, otherwise the plot is only shown
    sm_plots(models,save=True)
    
    #calculate the test log likelihoods and ELBOs
    tlls, elbos = [], []
    for model in models:
        tll = calculate_tll(model, X_test, y_test, y_sigma)
        tlls.append(tll)
        
        sess = model.enquire_session()
        elbo = np.mean([-sess.run(model.likelihood_tensor) for _ in range(10)])
        elbos.append(elbo)
        
    #The higher the better for both
    print(f'Test log likelihoods: {tlls[0]:.2f} (FC DGP), {tlls[1]:.2f} (STAR DGP), {tlls[2]:.2f} (MF DGP)')
    print(f'ELBOs: {elbos[0]:.1f} (FC DGP), {elbos[1]:.1f} (STAR DGP), {elbos[2]:.1f} (MF DGP)')