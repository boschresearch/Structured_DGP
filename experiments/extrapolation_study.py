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
            
import argparse
import os

from utils.models import prepare_model
from utils.prepare_data import prepare_dataset_extrapolate
from utils.optimise import run_adam
from utils.logger_io import save_tlls, get_tlls
from utils.evaluate_performance import direct_comparison

if __name__ == '__main__':
    
    #run or evaluate, or both?
    parser = argparse.ArgumentParser(description='Comparing MF and STAR DGP on extrapolation task on boston dataset')
    parser.add_argument('--run', help='run the experiment', action='store_true', default=False)
    parser.add_argument('--eval', help='evaluate the results', action='store_true', default=False)
    parser.add_argument('--seed', help='seed for random numbers', default=42, type=int)
    parser.add_argument('--seed_list_eval', help='allows to pass a list of seeds to evaluate, e.g. 0 21 42 3',
                        default=None, type=int, nargs="*")
    parser.add_argument('--batch_size', help='set the minibatch size for training', default = 512, type = int)
    parser.add_argument('--dataset', help='dataset name', default = 'energy', type = str)
    args = parser.parse_args()
    
    #some default settings
    seed_list_eval = args.seed_list_eval
    if args.seed_list_eval is None:
        seed_list_eval = range(10)
    seed = args.seed
    batch_size = args.batch_size
    dataset = args.dataset
    M = 128
    results_dir = 'results'    
    train_iters = 20000
    adam_lr = 0.005
    
    #run the experiment
    if args.run:
        #get data
        X, y, X_test, y_test, y_sigma, Z, holdout_list = prepare_dataset_extrapolate(seed = seed, M = M,
                                                                                    name = dataset)
        test_list = [X_test, y_test, y_sigma]
        
        #prepare models
        star_dgp = prepare_model('star', X, y, Z, minibatch = batch_size, M = M, small_architecture=False)
        mf_dgp = prepare_model('mf', X, y, Z, minibatch = batch_size, M = M, small_architecture=False)
        
        #train models
        _, time_mf, iters_mf = run_adam(mf_dgp,train_iters,adam_lr,holdout_list)
        print(f'Training the MF DGP for {iters_mf} '
              f'iterations took {time_mf:.2f}s \n')
        
        _, time_star, iters_star = run_adam(star_dgp,train_iters,adam_lr,holdout_list)
        print(f'Training the STAR DGP for {iters_star} iterations '
              f'took {time_star:.2f}s \n')
        
        #save the predictions for individual test points in a h5 file
        star_dgp_dict={'model':star_dgp, 'base_name':f'{dataset}_tll_star'}
        mf_dgp_dict = {'model':mf_dgp, 'base_name':f'{dataset}_tll_mf'}
        save_tlls(results_dir,star_dgp_dict, mf_dgp_dict, seed, test_list)
    
    #evaluate the experiments
    if args.eval:
        #prepare file names
        star_dict = {'base_name':os.path.join(results_dir,f'{dataset}_tll_star')}
        mf_dict = {'base_name':os.path.join(results_dir,f'{dataset}_tll_mf')}
        dicts = [star_dict,mf_dict]        
        
        #read the tlls        
        tlls_star_dict, tlls_mf_dict = get_tlls(dicts,seed_list_eval)
        
        #get the results of a direct comparison
        percentage_star_better, difference_star_better, common_seed_list = direct_comparison(tlls_star_dict, tlls_mf_dict)
        if len(common_seed_list) == 1:
            print(f'The STAR DGP outperformed the MF DGP on '
                  f'({percentage_star_better[0]:.2f} +- {percentage_star_better[1]})% '
                  f'of the test samples averaged over {len(common_seed_list)} seed.\n'
                  f'The average tll differences between STAR DGP and MF DGP were '
                  f'({difference_star_better[0]:.3f} +- {difference_star_better[1]}) ')
        else:
            print(f'The STAR DGP outperformed the MF DGP on '
                  f'({percentage_star_better[0]:.2f} +- {percentage_star_better[1]:.2f})% '
                  f'of the test samples averaged over {len(common_seed_list)} seeds.\n'
                  f'The average tll differences between STAR DGP and MF DGP were '
                  f'({difference_star_better[0]:.3f} +- {difference_star_better[1]:.3f}) ')
            
        
        