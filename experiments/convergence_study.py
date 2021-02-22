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
Script to reproduce the convergence study reported in Fig. 2 in the paper.
By default the uci concrete dataset is used.
"""

import argparse

from utils.models import prepare_model
from utils.prepare_data import prepare_dataset
from utils.optimise import run_adam
from utils.logger_io import save_logger, get_logger
from utils.plots import convergence_plot

if __name__ == '__main__':
    #run or plot, or both?
    parser = argparse.ArgumentParser(description='Running convergence comparison on boston dataset')
    parser.add_argument('--run', help='run the experiment', action='store_true', default=False)
    parser.add_argument('--plot', help='plot the results', action='store_true', default=False)
    args = parser.parse_args()
    
    #some default settings
    seed = 42
    M = 128
    results_dir = 'results'
    filenames = ['logger_convergence_analytical.h5','logger_convergence_sampled.h5']
    train_iters_ana = 750
    train_iters_sam = 1500
    adam_lr = 0.005
    
    #run the experiment
    if args.run:
        #get data
        X, y, _, _, _, Z = prepare_dataset(seed = seed, M = M, name='concrete')
        
        #prepare models
        fc_dgp = prepare_model('fc', X, y, Z.copy(), M = M, small_architecture=False)
        fc_dgp_sampled = prepare_model('fc_sampled', X, y, Z.copy(), M = M, small_architecture=False)
        
        #train models
        logger_analytical, time_analytical, iters_ana = run_adam(fc_dgp,train_iters_ana,adam_lr)
        logger_sampled, time_sampled, iters_sam = run_adam(fc_dgp_sampled,train_iters_sam,adam_lr)
        print(f'Training with analytical marginalisation for {iters_ana} '
              f'iterations took {time_analytical:.2f}s \n'
              f'Training with marginalisation by sampling for {iters_sam} '
              f'iterations took {time_sampled:.2f}s')
        
        #prepare dict for saving loggers
        files_and_loggers = {filenames[0]:logger_analytical, filenames[1]:logger_sampled}
        
        #save loggers as .h5 in results directory
        save_logger(results_dir, files_and_loggers) 
    
    #plot the results
    if args.plot:
        times, elbos = get_logger(results_dir,filenames)
        convergence_plot(times, elbos, filenames, save=True, results_dir=results_dir)
        
        
    
    
    