# %%

import math
import numpy as np
import sympy as sp
from scipy import special
from scipy.constants import hbar, e, epsilon_0

import matplotlib.pyplot as plt


# adding the ...QuDiPy to path

from pathlib import Path
import sys, os

p = Path(__file__).parents[2]
sys.path.append(str(p))


import qudipy.utils.constants as C 


def hl_exchange(dot_sep, e_field, omega0, material='vacuum'):
   
    consts = C.Constants(material)

    #material constants
    m = consts.me
    kappa = consts.epsR

    # Bohr radius
    a_B = np.sqrt(hbar/ (m * omega0))
    # dimensionless distance
    d = dot_sep/ a_B

    #dimensionless parameter
    c = (math.sqrt(math.pi / 2) * e**2 / 
            (4 * math.pi * epsilon_0 * kappa * a_B) / (hbar * omega0)
                )
                
    b=1
    Jhl = (hbar * omega0 / np.sinh(2 * d**2 *(2 * b - 1 / b))) * (c * np.sqrt(b) * 
            (np.exp(-b * d**2) * special.iv(0, (b * d**2)) 
                     - np.exp(d**2 * (b - 1 / b))* special.iv(0, (d**2 * (b - 1 / b))))
                                                    + 3 / (4 * b) * (1 + b * d**2))
    J_hl = (hbar * omega0 / np.sinh(2 * d**2)
                * (c * (np.exp(-d**2) * special.iv(0, d**2) -1 ) 
                    + 3 / 4 * (1 + d**2))
                ) 
    return Jhl, J_hl
# defining constants

if __name__=='main':

    c = 2.4

    consts = qudipy.constants.Constants('GaAs')

    #material constants
    m = consts.me
    kappa = consts.epsR
    
    print('Hello')


    #I know, it would be more logical to make 
    a_B = c / (math.sqrt( math.pi /2 ) * e**2 *m / 
                    (4 *math.pi * epsilon_0 * kappa * hbar**2))

    d_s = np.linspace(0.45,0.85,20)
    a_s = d_s * a_B

    omega_0 = hbar /  (m * a_B**2)

    J_HLs = hl_exchange(a_s, 0, omega_0)

    
    for j in J_HLs: 
        plt.plot(d_s, J_HLs)
      
    plt.show()
# %%
