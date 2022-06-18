from configparser import Interpolation
import os, sys

from imageio import save
sys.path.append(os.path.dirname(os.getcwd()))


import qudipy.potential as pot
import qudipy.exchange as ex
import qudipy.qutils as qt
import qudipy.utils.helpers as hp
import qudipy.starkshift as ss

from qudipy.utils.constants import Constants
from qudipy.potential import InterpolateND

import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm

csts = Constants('Si/SiO2')

data_exist = False
plot_flag = False

input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_dotsep_60nm copy')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_sweep_4')

output_preprocessed = os.path.join(sys.path[0],'tutorials', 'QuDiPy tutorial data','Pre-processed potentials','Pre-processed_data')

png_dir = os.path.join(sys.path[0],'data')

if not data_exist:
        
    # Importing potential
    potential = pot.process_nextnano.import_dir(input_nextnano, show_files=True)


    # Desired Slice
    z = -0.2
    _, nearest_slice = hp.find_nearest(potential[0]['coord']['z'], z)

    print(f'The nearest slice for z = {z} is: {nearest_slice}')
    # Specifying control values and names
    pot_dir = output_preprocessed + '_for_nearest_slice{:.3e}'.format(nearest_slice) + '/'

    # TODO: allow user to extract ctrl_values for only certain ctrl_names
    ctrl_names = ['GateContact_T1','GateContact_P1','GateContact_T2','GateContact_P2','GateContact_T3','GateContact_S','SiContact'  ]

    ctrl_vals = pot.process_nextnano.get_ctrl_vals(potential) # Must make sure that ctrl_vals are sorted

    # Now we define the field types we wish to write 2D slices for. Either potential or the electric field.
    pot.process_nextnano.write_data(input_nextnano,output_preprocessed, slice=z, f_type=['potential','field'])

    loaded_data_pot = pot.load_potentials(ctrl_vals, ctrl_names,
                                    f_type='pot', f_dir=pot_dir,
                                    f_dis_units='nm', f_pot_units='eV')

    loaded_data_field = pot.load_potentials(ctrl_vals, ctrl_names,
                                    f_type='electric', f_dir=pot_dir,
                                    f_dis_units='nm', f_pot_units='V/nm')


    gate = []
    for i in range(np.shape(loaded_data_pot['ctrl_vals'])[0]):
        gate.append(loaded_data_pot['ctrl_vals'][i])

    # Convert list of lists to array
    gate = np.array(gate)

    ctrl_vals = []
    for i in range(np.shape(gate)[1]):
        # Append all unique control values per control variable. This is done
        # by: converting array of voltages per gate to set, sort the set items,
        # convert the set to a list, and append the list to ctrl_vals.
        ctrl_vals.append(list(sorted(set(gate[:,i]))))
    
    # Determine data set size for array initialization
    tup = ()
    for i in ctrl_vals:
        tup += (len(i),)

    gl_data = np.empty(tup)
    gr_data = np.empty(tup)
    # gl_data[:] = np.NaN

    print(f'Processing ')
    
    # Loop over all combinations of contrl values
    for idx, param in enumerate(tqdm(loaded_data_pot['ctrl_vals'])):

        # Cacluating delta_g for N number of plunger gates and N+2 tunnel gates
        # --------------------------------------------------------------------------

        # Copy potential data for later use
        pot_T = loaded_data_pot['potentials'][idx].copy()
        pot_L = loaded_data_pot['potentials'][idx].copy()
        pot_R = loaded_data_pot['potentials'][idx].copy()

        # Define gridparameters for dot array and subdot array
        _2d_coords = loaded_data_pot['coords']
        x = _2d_coords.x
        y = _2d_coords.y

        gparams = pot.GridParameters(x,y)
        gparamsl = pot.GridParameters(x,y)
        gparamsr = pot.GridParameters(x,y)

        # Determine index in n-dimensional data arrays for later data assignment
        idx = []
        for n,i in enumerate(param):
            idice = np.where(ctrl_vals[n] == i)
            idx.append([int(idice[0][0])])

        idex = np.ix_(*idx)

        # Right dot ------------------------------------------

        # Restrict x-coordinates to right dot
        x_r = x[x>0]
        idx_r = len(x_r)
        n,m = np.shape(pot_R)
        base = np.amax(pot_R)
        pot_R[:,:idx_r] = base
        gparamsr.update_potential(pot_R)

        _, e_vecsr = qt.solvers.solve_schrodinger_eq(csts, gparamsr, n_sols=1)
        wfr = e_vecsr[:,:,0]

        # g-factor
        stark_shiftr = ss.starkshift.StarkShift(gparamsr, csts)
        dummy_e_intr = lambda V: loaded_data_field['electric'][np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]
        dgr = stark_shiftr.delta_g(dummy_e_intr, [param], ctrl_names,
                                                        wavefuncs=[wfr])['delta_g_1'][0]
        
        # Assign delta-g to appropriate coordinate in n-dimensional data set
        gr_data[idex] = dgr

        # Left dot ------------------------------------------

        # Restrict x-coordinates to left dot
        x_l = x[x<0]
        idx_l = len(x_l)
        n,m = np.shape(pot_L)
        base = np.amax(pot_L)
        pot_L[:,idx_l:] = base

        gparamsl.update_potential(pot_L)

        _, e_vecsl = qt.solvers.solve_schrodinger_eq(csts, gparamsl, n_sols=2)
        wfl = e_vecsl[:,:,0]

        # g-factor
        stark_shiftl = ss.starkshift.StarkShift(gparamsl, csts)
        dummy_e_intl = lambda V: loaded_data_field['electric'][np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]
        dgl = stark_shiftl.delta_g(dummy_e_intl, [param], ctrl_names,
                                                        wavefuncs=[wfl])['delta_g_1'][0]

        # Assign delta-g to appropriate coordinate in n-dimensional data set
        gl_data[idex] = dgl

        # Calculate exchange
        # --------------------------------------------------------------------------
        
        gparams.update_potential(pot_T)

        ne = 1
        n_xy_ho = (4,4)
        n_se_orbs = 2
        ommega_guess = omega_guess = 0.025 * csts.hbar / (csts.me * csts.a_B**2)
        data_dir = ''

        ens, __ = qt.solvers.solve_many_elec_SE(gparams, n_elec=ne,
                    n_xy_ho=n_xy_ho,
                    n_se=n_se_orbs,
                    n_sols=4,
                    consts=csts,
                    optimize_omega=False,
                    omega=omega_guess,
                    cme_dir= data_dir,
                    spin_subspace=[0])
        J = ens[1] - ens[0]
        print('exchange value is: ', J,
        '\nenergies are: ', ens)
        ex.append(J)
    
    f = open("ctrl_vals.pkl","wb")
    # write the python object (dict) to pickle file
    pk.dump(ctrl_vals,f)
    f.close()

    f = open("gl_data.pkl","wb")
    # write the python object (dict) to pickle file
    pk.dump(gl_data,f)
    f.close()


    f = open("gr_data.pkl","wb")
    # write the python object (dict) to pickle file
    pk.dump(gr_data,f)
    f.close()
else:
    file_to_read = open("ctrl_vals.pkl", "rb")
    ctrl_vals = pk.load(file_to_read)
    
    file_to_read = open("gl_data.pkl", "rb")
    gl_data = pk.load(file_to_read)

    file_to_read = open("gr_data.pkl", "rb")
    gl_data = pk.load(file_to_read)




# 3D -------------------------------------------------------------------

# Build coordinates for interpolation
COR = []
for i in range(len(ctrl_vals)):
    # COR.append(np.linspace(min(ctrl_vals[i]), max(ctrl_vals[i]), len(ctrl_vals[i])))
    COR.append(np.linspace(min(ctrl_vals[i]), max(ctrl_vals[i]), 20))

# gl interpolation -----------------------------

# Create inperaltion object for delta-g in left dot
interpl = InterpolateND(gl_data, *ctrl_vals)

# Generate delta-g values for interpolation corrdinates
interpl(*COR)

# Vizulize data
interpl.Crossection2D(phys_param='gl')
interpl.Crossection3D(COR, phys_param='gl')


# gr interpolation -----------------------------

# # Create inperaltion object for delta-g in left dot
# interpr = InterpolateND(gr_data, *ctrl_vals)

# # Generate delta-g values for interpolation corrdinates
# interpr(*COR)

# # Vizulize data
# interpr.Crossection2D(phys_param='gr')
# interpr.Crossection3D(COR, phys_param='gr')

plt.show()





