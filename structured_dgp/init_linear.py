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

from gpflow.mean_functions import Zero, Identity, Linear
import numpy as np

"""
The following function is adapted from Doubly-Stochastic-DGP V 1.0
( https://github.com/ICL-SML/Doubly-Stochastic-DGP/ 
Copyright 2017 Hugh Salimbeni, licensed under the Apache License 2.0,
cf. 3rd-party-licenses.txt file in the root directory of this source tree).
"""

def init_linear(X,Z,all_kernels,initialized_Zs=False):
    """
    if there are no Zs from an initialization (e.g. for warm-starting),
    all_Zs is initialized according to the Salimbeni scheme (Z should be MxD).
    otherwise the Zs obtained from the initialization are simply taken and put
    into the all_Zs array (Z should be a list of L arrays)
    """
    if initialized_Zs:
        all_Zs = Z
    else:
        all_Zs = []
    all_mean_funcs = []
    X_running = X.copy()
    if not initialized_Zs:
        Z_running = Z.copy()
    for kern_in, kern_out in zip(all_kernels[:-1], all_kernels[1:]):
        dim_in = kern_in.input_dim
        dim_out = kern_out.input_dim
        if dim_in == dim_out:
            mf = Identity()
        else:
            if dim_in > dim_out:  # stepping down, use the pca projection
                _, _, V = np.linalg.svd(X_running, full_matrices=False)
                W = V[:dim_out, :].T
                
            else: # stepping up, use identity + padding
                W = np.concatenate([np.eye(dim_in), np.zeros((dim_in, dim_out - dim_in))], 1)
                
            mf = Linear(W)
            mf.set_trainable(False)
        
        all_mean_funcs.append(mf)
        if not initialized_Zs:
            all_Zs.append(Z_running)

        if dim_in != dim_out:            
            X_running = X_running.dot(W)
            if not initialized_Zs:
                Z_running = Z_running.dot(W)

    # final layer    
    all_mean_funcs.append(Zero())
    if not initialized_Zs:
        all_Zs.append(Z_running)
    return all_Zs, all_mean_funcs