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
The following classes and utility functions are adapted from GPflow V 1.5.1
( https://github.com/GPflow/GPflow/tree/develop-1.0 
Copyright 2018 GPflow authors, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree),
more specifically from the demo IPython notebook at
https://github.com/GPflow/GPflow/blob/develop-1.0/doc/source/notebooks/advanced/gps_for_big_data.ipynb
"""

import numpy as np
import tensorflow as tf

import time

from scipy.stats import norm #normal distribution
from scipy.special import logsumexp #stable implementation of log sum exp

import gpflow
from gpflow.actions import Action

class Logger(Action):
        
    """
    Basic logger (monitoring) class that tracks time and elbo (optimisation objective)
    during the optimisation.
    If a holdout data set is given, the performance (in test log likelihood) on that
    validation set is regularly tracked during the optimisation.
    This is required for early stopping.
    """

    def __init__(self, model, holdout_list=None):
        self.model = model
        self.elbo_log = []
        self.time_log = []
        
        self.X_holdout = None
        if holdout_list is not None:
            self.X_holdout, self.y_holdout, self.sigma = holdout_list
            self.tll_holdout_log = []
    
    def run(self, sess):
        likelihood = - sess.run(self.model.likelihood_tensor)
        self.elbo_log.append(likelihood)
        self.time_log.append(time.time())
        if self.X_holdout is not None:
            self.tll_holdout_log.append(self.predict(self.X_holdout,self.y_holdout))
    
    def predict(self, X_test, y_test, y_sigma=None, num_samples=5):
        if y_sigma is None:
            y_sigma = self.sigma
        
        mean, var = self.model.predict_y(X_test, num_samples)        

        # test log-likelihood (with rescaled output values), calculated numerically more stable
        tll_SN = np.sum(norm.logpdf(y_sigma*y_test, loc = y_sigma*mean,
                                    scale = y_sigma*var**0.5), 2)
        tll = logsumexp(tll_SN, 0, b=1/float(num_samples))
        
        return np.average(tll)

def run_adam(model, iterations, learning_rate,
            holdout_list=None, filtersize=50):
    
    """
    Utility function providing more control over the adam optimizer.
    Implements exponential decay of the learning rate and early stopping (optional).
    Also uses the Logger class defined above to track the optimisation progress,
    which is also needed for early stopping.
    """
    early_stopping=False
    if holdout_list is not None:
        early_stopping = True
    
    iterations_info = iterations

    logger = Logger(model, holdout_list=holdout_list)
    
    sess = model.enquire_session()

    #exponentially decaying learning rate, damp with a factor of 0.98 every 1000 iterations
    global_step = tf.Variable(0, trainable=False)
    op_increment = tf.assign_add(global_step, 1)
    lr = tf.cast(tf.train.exponential_decay(learning_rate, global_step, 1000, 0.98,
                                            staircase=True), dtype=tf.float64)
    
    # create the optimisation op.
    # note that the option colocate_gradients_with_ops is needed for forcing
    # execution of backward pass computations on the same device as the forward
    # pass computations occured. this is important since some computations have
    # to performed on a cpu for efficiency (see ../../structured_dgp/all_layers.py)
    sess.run(tf.variables_initializer([global_step]))
    opt = gpflow.train.AdamOptimizer(lr)
    op_train = opt.make_optimize_tensor(model, colocate_gradients_with_ops=True)
    
    start = time.time()
    tll_smoothed = np.zeros(iterations)
    for it in range(iterations):
        
        if it % 10 == 0:
            logger.run(sess)
        sess.run(op_train)
        sess.run(op_increment)

        # early-stopping criterion
        converged = False
        if early_stopping:
            tll_smoothed[it] = np.mean(logger.tll_holdout_log[-filtersize:])
            
            for k in range(5):
                if  (it>5000) and (tll_smoothed[it-k*filtersize] < tll_smoothed[it-(k+1)*filtersize]):
                    converged = True
                    iterations_info = it
                    pass
                else:
                    converged = False
                    break

        if converged:
            break

    opt_time = time.time() - start
    model.anchor(model.enquire_session())
    return logger, opt_time, iterations_info
