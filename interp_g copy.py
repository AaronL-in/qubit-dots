import numpy as np
from itertools import product

import qudipy as qd
from qudipy.potential import GridParameters
from qudipy.potential.potential_interpolator import PotentialInterpolator

def interp_g(load_data_dict, g_idx='gl', x='pl', y='pr', z='t'):

    # Get first set of x, y, and z coordinates
    x_coords = load_data_dict[x]
    y_coords = load_data_dict[y]
    z_coords = load_data_dict[z]

    # Extract all the control values
    all_ctrls = np.asarray(load_data_dict[g_idx])
    
    # Get total number of ctrls (including singleton dims)
    n_ctrls = len(load_data_dict[g_idx][0])
        
    # Find which gate voltages have singleton dimension. We need to keep track
    # because the interpolator cannot handle singleton dimensions
    single_dims = []
    n_dims = []
    ctrl_values = []
    for idx in range(n_ctrls):
        n_unique = len(set(all_ctrls[:,idx]))
        if n_unique == 1:
            single_dims.append(idx)
        else:
            n_dims.append(n_unique)
            ctrl_values.append(sorted(list(set(all_ctrls[:,idx]))))
        
    # Now assemble the data to be interpolated
    temp_n_dims = [range(n) for n in n_dims]
    
    # Add the y and x coordinate lengths so we know the expected dimensions of 
    # the total nd array of data to interpolate
    if y_slice is None:
        all_data_stacked = np.zeros((np.prod(n_dims),len(y_coords),len(x_coords)))
        n_dims.extend([len(y_coords),len(x_coords)])  
    else:
        all_data_stacked = np.zeros((np.prod(n_dims),len(x_coords)))
        n_dims.extend([len(x_coords)])  

    # Go and stack the potential data together and then reshape it into
    # correct format
    if y_slice is not None:
        y_idx = qd.utils.find_nearest(y_coords, y_slice)[0]
    for idx, curr_gate_idx in enumerate(product(*temp_n_dims)):
        if y_slice is None:
            all_data_stacked[idx,:,:] = load_data_dict[f_type][idx]
        else:
            all_data_stacked[idx,:] = np.squeeze(
                load_data_dict[f_type][idx][y_idx,:])
    
    all_data_stacked = np.reshape(all_data_stacked,(n_dims))
    
    # Construct the interpolator
    if y_slice is None:
        ctrl_values.extend([y_coords,x_coords])
    else:
        ctrl_values.extend([x_coords])
    interp_obj = PotentialInterpolator(ctrl_values, load_data_dict['ctrl_names'],
                                        all_data_stacked, single_dims, constants,
                                        y_slice)
                                    
    return interp_obj