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
    
import numpy as np
import time

from gpflow.training import AdamOptimizer

from utils.plots import covariance_plots
from utils.prepare_data import prepare_dataset
from utils.evaluate_performance import calculate_tll
from utils.models import prepare_model

if __name__ == "__main__":
    
    #prepare the data and the inducing points
    M = 30 #number of inducing points
    X, y, X_test, y_test, y_sigma, Z = prepare_dataset(seed = 42, M = M)
    
    
    #Initialize three models, FC, STAR, and MF DGP with the same architecture,
    #three layers with 2 GPs in the first and second layer, respectively,
    #and one GP in the final layer.
    model_names = ['fc','star','mf']
    models = [prepare_model(name, X, y, Z, M = M) for name in model_names]
    
    #train the models
    lr = 0.015
    for model, model_name in zip(models,model_names):
        print(f'Training {model_name} model')
        time_start= time.time()
        opt = AdamOptimizer(lr)
        opt.minimize(model,maxiter=4000,colocate_gradients_with_ops=True)
        timing = time.time()-time_start
        print(f'Training took {timing:.2f}s')
    #plot the covariance matrices
    #if save is True, a .png is saved in the same directory, otherwise the plot is only shown
    covariance_plots(models,save=True,results_dir='results')
    
    #calculate the test log likelihoods and ELBOs
    tlls, elbos = [], []
    for model in models:
        tll = np.average(calculate_tll(model, X_test, y_test, y_sigma))
        tlls.append(tll)
        
        sess = model.enquire_session()
        elbo = np.mean([-sess.run(model.likelihood_tensor) for _ in range(100)])
        elbos.append(elbo)
        
    #The higher the better for both
    print(f'Test log likelihoods: {tlls[0]:.2f} (FC DGP), {tlls[1]:.2f} (STAR DGP), {tlls[2]:.2f} (MF DGP)')
    print(f'ELBOs: {elbos[0]:.1f} (FC DGP), {elbos[1]:.1f} (STAR DGP), {elbos[2]:.1f} (MF DGP)')