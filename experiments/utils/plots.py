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
Script contains two plotting utilities. One for the covariance matrices used in demo_boston,
the other for the convergence study reported in the paper.
"""

import numpy as np
import os

import matplotlib.pyplot as plt

def covariance_plots(models,save=False,results_dir=None):
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
        if results_dir is None:
            plt.savefig('covariance.png',quality=95,dpi=400,bbox_inches = "tight")
        else:
            plt.savefig(os.path.join(results_dir,'covariance.png'),quality=95,dpi=400,bbox_inches = "tight")
        plt.close()
    else:
        plt.show()
    
    return

def convergence_plot(times, elbos, filenames, save=False, results_dir=None):
    """
    Given the loggers of times and elbos, of the analytical and MC sampling marginalisation methods,
    create the plot showing the curves as a function of time.
    """
    #settings for the plot
    EXTRA_SMALL_SIZE = 12
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=EXTRA_SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    fig = plt.figure(figsize = (3,3))
    ax = fig.add_subplot(1,1,1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    colors = {filenames[0]:'#D8A6FF',
              filenames[1]:'#16B857'}
    labels = {filenames[0]:'Analytical (ours)',
              filenames[1]:'MC sampling'}
    
    for key in times.keys():
        plt.plot(times[key],elbos[key],label=labels[key],color = colors[key],linewidth=3)
    plt.legend(loc=4)
    plt.xlabel('time [sec.]')
    plt.ylabel('ELBO [arb. unit]')
    plt.xlim([-2,times[filenames[1]]])
    
    if save:
        if results_dir is None:
            plt.savefig('convergence.png',quality=95,dpi=400,bbox_inches = "tight")
        else:
            plt.savefig(os.path.join(results_dir,'convergence.png'),quality=95,dpi=400,bbox_inches = "tight")
        plt.close()
    else:
        plt.show()
        
    return