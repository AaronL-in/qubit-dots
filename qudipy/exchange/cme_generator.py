#this file generates a library of CMEs


import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import qudipy as qd
import qudipy.potential as pot
import qudipy.exchange as exch
#from scipy.signal import find_peaks    # for finding local minima

#from tqdm import tqdm
import time
import math
import numpy as np
import copy
import gc       # for memory cleanup

#required if parallelization is requested
#rom numba import njit, boolean, int32, double, prange

omega = 7.53E+12        #value used from one of the optimization results
import gc

durations = {}
for n in range(12,19,2):
    temp = time.time()
    matrix = exch.calc_origin_cme_matrix(n, n, omega=omega, consts=qd.Constants("GaAs"), 
                           rydberg=False,  # parallelize=True,
                            save_dir='D:\\QuDiPy tutorial data\\CMEs',
                           #save_dir='..\\tutorials\\QuDiPy tutorial data\\CMEs',
                           #save_dir=None
                           )
    durations['n='+str(n)]= time.time() - temp
    del matrix
    del temp
    gc.collect()
    
print(durations)