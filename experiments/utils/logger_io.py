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
Input/output utility functions for the loggers that are created during training
and the test log likelihoods at testing time.
"""

import numpy as np
import h5py
import os

from utils.evaluate_performance import calculate_tll

def save_logger(results_dir, filenames_and_loggers):
    
    #result directory
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
            
    #write info for both marginalisations
    for key in filenames_and_loggers.keys():
        f = h5py.File(os.path.join(results_dir, key),'w')
        f_out = f.create_group('logger')
        f_out['elbo'] = filenames_and_loggers[key].elbo_log
        f_out['time'] = filenames_and_loggers[key].time_log
        f.close()
    return
    
    
def get_logger(results_dir, filenames):
    path_in = results_dir
    results_time = {}
    results_elbo = {}
    for file in filenames:
        f = h5py.File(os.path.join(path_in, file),'r')
        times = np.array(f['logger']['time'])
        times = times - times[0]
        elbo = -np.array(f['logger']['elbo'])/10000
        results_time[file] = times
        results_elbo[file] = elbo
        f.close()
    
    return results_time, results_elbo

def save_tlls(results_dir, star_dict, mf_dict, seed, test_list):
    
    #result directory
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    
    #save tlls as h5 files      
    for dic in [star_dict, mf_dict]:
        path = os.path.join(results_dir,f"{dic['base_name']}_seed{seed}.h5")
        tlls = calculate_tll(dic['model'],*test_list,num_samples=1000)
        f = h5py.File(path,'w')
        f_out = f.create_group('predictions')
        f_out['tlls'] = np.array(tlls)
        f.close()
        
    return

def get_tlls(dicts,seeds):
    success = False
    for dic in dicts:
        for seed in seeds:            
            path = f'{dic["base_name"]}_seed{seed}.h5'
            print(f'Trying to read file {path}')
            if os.path.exists(path):
                success = True
                file = h5py.File(path,'r')
                tlls = np.array(file['predictions']['tlls'])
                dic[f'tlls_seed{seed}'] = tlls
                file.close()
    assert success, 'No file with the provided seeds found'   
    return dicts




