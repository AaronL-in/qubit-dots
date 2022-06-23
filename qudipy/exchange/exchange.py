'''
Exchange calculation for a potential
'''

import numpy as np
from scipy import special
from scipy.constants import hbar, e, epsilon_0

import matplotlib.pyplot as plt

# adding the ...QuDiPy to path
from pathlib import Path
import sys,os

p = Path(__file__).parents[2]
sys.path.append(str(p))

from qudipy.utils import Constants
import qudipy.potential as pot


def hl_quartic(gparams,  y_slice=0, material='vacuum'):
    '''
    Function calculates exchange values in Heitler-London approximation
    for an array of dots. Each pair is fitted to a quartic function for the
    calculation.
    Formula borrowed from Burkard & Loss, PRB 89 (3), 1999, pgs. 2073-2074  

    Parameters:
    -------------
    
    gparams: GridParameters Object
        Must contain 2D potential, and x and y coordinates as a meshgrid object


    Keyword Arguments:
    -------------------
    y_slice: Int, Function, optional 
        The y-value (or x-dependent function) of the 
        y_slice where the dots are located. Default is 0.

    #TODO find out where y-value being a function can be used

    material: String, optional
        The string defining the material that the dots are
        in. Default is 'vacuum'.

    Returns:
    --------------
    J: float or tuple of floats
        Exchange value(s) for the array of dots
    '''
    
    #material constants
    consts = Constants(material)
    m = consts.me
    kappa = consts.kappa

    #obtaining parameters from fitting to a quartic potential
    fit_params = pot.manipulate.fit_quartic(gparams, y_slice=y_slice, 
                                        material=material, return_params=True)
    a = fit_params['dot_sep'] / 2
    e_field = fit_params['e_field']
    omega0 = fit_params['omega0']

    # Bohr radius
    a_B = np.sqrt(hbar/ (m * omega0))
    # dimensionless distance
    d = a/ a_B
    #dimensionless parameter
    c = a_B * (np.sqrt( np.pi /2 ) * e**2 *m / 
                    (4 *np.pi * epsilon_0 * kappa * hbar**2))
      
    J_hl = ((hbar * omega0 / np.sinh(2 * d**2)
                * (c * (np.exp(-d**2) * special.iv(0, d**2) -1 ) 
                    + 3 / 4 * (1 + d**2))) 
                        + (1 / np.sinh(2 * d**2)) * (3 / (2 * d**2)) *
                            (e * e_field * a )**2 / (hbar * omega0)
                )
    return J_hl
# defining constants

if __name__=='__main__':
    # Define the source directory for the nextnano3D simulation data.
    input_nextnano =    os.path.join('d:\\', 'DQD_dotsep_60nm_v2')
    # Define a directory location which will contain the pre-processed potential
    # data for a given 2D slice. If this directory doesn't exist currently then 
    # it will be created.
    output_preprocessed = os.path.join(sys.path[0], 'Test_Pre-processed_data')

    print(input_nextnano)
    print(output_preprocessed)

    z = -0.2
    potential = pot.process_nextnano.import_dir(input_nextnano, show_files=False)
    
    # save_flag = pot.process_nextnano.write_data(input_nextnano,
    #                                 output_preprocessed,
    #                                     slice=z, f_type=['potential','field'])

    pot_dir = output_preprocessed + '_for_nearest_slice{:.3e}'.format(z)

    ctrl_vals = pot.process_nextnano.get_ctrl_vals(potential)

    ctrl_names = ('GateContact_T1', 'GateContact_P1', 'GateContact_T2', 
            'GateContact_P2', 'GateContact_T3', 'GateContact_S', 'SiContact')

        
    loaded_data = pot.load_potentials(ctrl_vals, ctrl_names,
                                    f_type='pot', f_dir=pot_dir,
                                    f_dis_units='nm', f_pot_units='V')

    print(loaded_data.keys())

    fig = plt.figure(figsize=(12,8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    
    # potential along x-axis at y-axis slice
    counter = 0 
    for i in range(np.shape(loaded_data['coords'][1])[0]):
        ax1.plot(loaded_data['coords'][0].T/1E-9,loaded_data['potentials'][0][i,:])
        counter += 1
    ax1.set(xlabel='x-coords [nm]', ylabel='1D potential [J]',
    title=f'Potentials along x-axis')  
    ax1.grid() 
    print(np.shape(loaded_data['coords'][0]))

    plt.show()