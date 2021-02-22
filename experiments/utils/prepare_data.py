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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

import pandas

from sklearn.datasets import load_boston

from scipy.cluster.vq import kmeans2

def prepare_dataset(seed = None, M = 30, name = 'boston'):
    """
    Load and prepare either the boston, energy, or concrete uci data set:
    split it in a train and test set, normalize the data according to the training data,
    and finally initialize the inducing points.
    Inputs are a seed for a reproducible split and the number of inducing points M.
    """
    #random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
    
    #load the data set
    assert name in ['boston','energy','concrete'],f'Dataset {name} is not known'
    
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    energy_url = f'{base_url}00242/ENB2012_data.xlsx'
    concrete_url = f'{base_url}concrete/compressive/Concrete_Data.xls'
    
    if name == 'boston':
        X, y = load_boston(return_X_y=True)
    elif name=='energy':        
        data = pandas.read_excel(energy_url).values
        data = data[:,:-1] #strip the second output
        X, y = data[:,:-1], data[:,-1]
    elif name=='concrete':        
        data = pandas.read_excel(concrete_url).values
        X, y = data[:,:-1], data[:,-1]
        
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

def prepare_dataset_extrapolate(seed = None, M = 30, name = 'boston'):
    """
    Load and prepare either the boston, energy, or concrete uci data set:
    split it in a train and test set, normalize the data according to the
    training data, and finally initialize the inducing points.
    Inputs are a seed for a reproducible split and the number of inducing points M.
    """
    #random seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        tf.random.set_random_seed(seed)
    
    #load the data set
    assert name in ['boston','energy','concrete'],f'Dataset {name} is not known'
    
    base_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/'
    energy_url = f'{base_url}00242/ENB2012_data.xlsx'
    concrete_url = f'{base_url}concrete/compressive/Concrete_Data.xls'
    
    if name == 'boston':
        X, y = load_boston(return_X_y=True)
    elif name=='energy':        
        data = pandas.read_excel(energy_url).values
        data = data[:,:-1] #strip the second output
        X, y = data[:,:-1], data[:,-1]
    elif name=='concrete':        
        data = pandas.read_excel(concrete_url).values
        X, y = data[:,:-1], data[:,-1]
    
    #create random axis on which to project the data using the normalized inputs
    idx = X.std(axis=0)>1e-3 #only normalize features with non-zero variance
    X_std = (X[:,idx] - X[:,idx].mean(axis=0))/X[:,idx].std(axis=0)
    nFeatures = X_std.shape[1]
    weights = np.random.randn(nFeatures, 1)
    U = np.dot(X_std, weights)
    
    #then sort inputs according to their projection along that axis
    proj_order = np.argsort(U[:,-1])
    X, y = X[proj_order], y[proj_order]
        
    #split data along axis with 50/50 split
    start = int(0.5*X.shape[0])
    X_test, y_test = X[:start], y[:start]
    X, y = X[start:], y[start:]
    
    #remove features from the training set that are constant (leading to errors when normalising)
    idx = np.std(X, axis=0) > 1e-3
    X = X[:, idx]
    X_test = X_test[:,idx]
    
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
    
    #prepare holdout data set for early stopping   
    #take 10% of training data to determine early stopping        
    n_train = X_std.shape[0]
    n_holdout = int(0.1 * n_train)        
    idx_holdout = np.zeros(n_train, dtype=np.bool)        
    idx_holdout[np.random.randint(0, n_train, n_holdout)] = 1
    X_train = X_std[~idx_holdout]
    y_train = y_std[~idx_holdout]
    X_holdout = X_std[idx_holdout]
    y_holdout = y_std[idx_holdout]
    holdout_list = [X_holdout, y_holdout, y.std()]
    
    return X_train, y_train, X_test, y_test, y.std(), Z, holdout_list

