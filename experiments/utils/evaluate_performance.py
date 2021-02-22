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

from scipy.stats import norm #normal distribution
from scipy.special import logsumexp #stable implementation of log sum exp

def calculate_tll(model, X_test, y_test, y_sigma, num_samples=100, test_batches=10):
    """
    Evaluate the trained models on the test data set using the (average) test
    log likelihood as a metric. For a Gaussian likelihood it can be calculated
    in a numerically stable way as
    \frac{1}{N} \sum_{n=1}^N \log \sum{s=1}^S \exp \log \mathcal{N} (y_n|f_{n,s}^L,(\sigma_{n,s}^L)^2),
    where we sample every test point S (=num_samples) times through the DGP.
    If the number of test samples is larger than 10, they are minibatched using
    a batch size of test_batches (which is 10 by default).
    """
    if num_samples <= 10:
        test_batches = num_samples
    num_datapoints, num_tasks  = y_test.shape
    mean = np.zeros((num_samples, num_datapoints, num_tasks))
    var  = np.zeros((num_samples, num_datapoints, num_tasks))
    i=0
    while i < num_samples:
        mean[i:i+test_batches], var[i:i+test_batches] = model.predict_y(X_test, test_batches)
        i += test_batches
    
    # test log-likelihood (with rescaled output values), calculated numerically more stable
    tll_SN = np.sum(norm.logpdf(y_sigma*y_test, loc = y_sigma*mean, scale = y_sigma*var**0.5), 2)
    tll = logsumexp(tll_SN, 0, b=1/float(num_samples))
    
    return tll

def direct_comparison(tlls_star_dict, tlls_mf_dict):
    
    #get list of seeds that exists in both dictionaries
    matching_list = []
    for key_s in tlls_star_dict:        
        if key_s == 'base_name':
            continue
        for key_m in tlls_mf_dict:
            if key_m == key_s:
                matching_list.append(key_m)
    #strip the elements in matching_list down to their seed
    seed_list = [matching_item.split('_')[-1] for matching_item in matching_list]
    
    #do the comparisons
    percentages, differences = [], []
    for match in matching_list:
        tll_star = tlls_star_dict[match]
        tll_mf = tlls_mf_dict[match]
        #percentage of test samples on which star outperforms mf on one particular seed
        #print(tll_star)
        #print(tll_mf)
        percentages.append(np.mean(tll_star > tll_mf))
        #average absolute value that the star tll is better than the mf tll on one particular seed
        differences.append(np.mean(tll_star - tll_mf))
    
    if len(matching_list) != 1:
        percentage_star_better = [100*np.mean(percentages), 100*np.std(percentages)/np.sqrt(len(matching_list))]
        average_tll_star_better = [np.mean(differences), np.std(differences)/np.sqrt(len(matching_list))]
    else:
        percentage_star_better = [100*percentages[0], 'NA']
        average_tll_star_better = [differences[0], 'NA']
    
    return percentage_star_better, average_tll_star_better, seed_list