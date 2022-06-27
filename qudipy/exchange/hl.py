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


import qudipy.utils.constants as con


def hl_exchange(dot_sep, e_field, omega0, material='vacuum'):
   
    consts = con.Constants(material)

    #material constants
    m = consts.me
    kappa = 13.1

    # Bohr radius
    a_B = np.sqrt(hbar/ (m * omega0))
    # dimensionless distance
    d = dot_sep/ a_B

    #dimensionless parameter
    c = a_B * (math.sqrt( math.pi /2 ) * e**2 *m / 
                    (4 *math.pi * epsilon_0 * kappa * hbar**2))

    print('c=',c)
          

    J_hl = (hbar * omega0 / np.sinh(2 * d**2)
                * (c * (np.exp(-d**2) * special.iv(0, d**2) -1 ) 
                    + 3 / 4 * (1 + d**2))
                )  / 1.6e-22 
    return J_hl
# defining constants

if __name__=='__main__':

    c = 2.42


    consts = con.Constants('GaAs')

    #material constants
    m = consts.me
    kappa = 13.1

    #I know, it would be more logical to make 
    a_B = c / (np.sqrt( math.pi /2 ) * e**2 * m / 
                    (4 *math.pi * epsilon_0 * kappa * hbar**2))

    d_s = np.linspace(0.45,0.85,20)
    a_s = d_s * a_B

    omega_0 = hbar /  (m * a_B**2)

    print('First c=', c)

    J_HL = hl_exchange(a_s, 0, omega_0, 'GaAs')

    plt.figure(figsize=(6, 4))

    plt.plot(d_s, J_HL, 'or')
    


    plt.show()

