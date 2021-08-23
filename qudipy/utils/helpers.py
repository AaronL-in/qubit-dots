"""
General helper utilities

@author: simba
"""

import numpy as np
import math
from numpy.core.defchararray import zfill

from numpy.testing._private.utils import tempdir
        
def find_nearest(array, value):
    '''
    Function to find the closest value to a number in a given array.
    If array contains duplicate values of the nearest number, then first
    instance in the array will be returned.
    
    Parameters
    ----------
    array : ND float array-like object
        An ND array type object of floats.
    value : float
        Value which we want to find the closest number to in the array.

    Returns
    -------
    near_idx : tuple of ints
        Tuple of indices for the nearest value in array.
    near_value : float
        Nearest value in array.

    '''
    
    # Convert to numpy array if not already
    array = np.asarray(array)
    
    # Obtain the indices corresponding to which element in array is closest 
    # to value
    near_idx = np.unravel_index((np.abs(array - value)).argmin(), array.shape)
        
    # Return also the nearest value
    near_value = array[near_idx]
    
    return near_idx, near_value

def quick_sort(array, precision='.15f'):
    '''
    Function which performs a quick sort algorithm on a list of floats. A 
    user can define the precision of the algorithm with regard to float 
    comparisions for equality.
    
    Parameters
    ----------
    array : array
        A 1D float array which is not sorted.

    Keyword Arguments
    -----------------
    precision : string
        A string which defines the order of magnitude for epsilon. Note, that
        the integer in precision='.15f' represents an epsilon of order 1e-16.

    Returns
    -------
    array : array
        The 1D float array which has been sorted.
    '''

    low = []
    equal = []
    high = []

    epsilon = float(format(0.0, precision) + '1')

    # array contains more then one element
    if len(array) > 1:
        pivot = array[0]
        for x in array:
            # x is less than pivot and x != pivot for the choosen epsilon
            if (x - pivot) < 0 and abs(x - pivot) > epsilon:
                low.append(x)
            # x = pivot for the choosen epsilon
            elif abs(x - pivot) <= epsilon:
                equal.append(x)
            # x is greater than pivot and x != pivot for the choosen epsilon
            elif (x - pivot) > 0 and abs(x - pivot) > epsilon:
                high.append(x)
        return quick_sort(low)+equal+quick_sort(high)
    # array contains one element
    else:
        return array


