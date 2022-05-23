import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import qudipy.circuit as circ
import qudipy.potential as pot
import qudipy.qutils as qt
import qudipy.utils.helpers as hp

import qudipy.starkshift as ss
from qudipy.utils.constants import Constants

from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt

csts = Constants('Si/SiO2')

# pot_dir = os.path.join(sys.path[0], 'QuDiPy tutorial data','Nextnano simulations','TMPLATE_5Gate_1.358E15_noRGrind')

# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','TMPLATE_5Gate_1.358E15_noRGrind')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','2gate_t1_d_sep_10')
input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_dotsep_60nm')

output_preprocessed = os.path.join(sys.path[0],'tutorials', 'QuDiPy tutorial data','Pre-processed potentials','Pre-processed_data')

#importing potential
potential = pot.process_nextnano.import_dir(input_nextnano, show_files=True)

# Desired Slice
z = 0.45

_, nearest_slice = hp.find_nearest(potential[0]['coord']['z'], z)

print(f'The nearest slice for z = {z} is: {nearest_slice}')
#specifying control values and names
pot_dir = output_preprocessed + '_for_nearest_slice{:.3e}'.format(nearest_slice) + '/'

# TODO: allow user to extract ctrl_values for only certain ctrl_names
# ctrl_names = ['V1','V2','V3','V4','V5']
ctrl_names = ['V_P','VTUN','BIAS']
ctrl_names = ['V_P','VTUN','BIAS']
ctrl_names = ['GateContact_T1','GateContact_P1','GateContact_T2','GateContact_P2','GateContact_T3','GateContact_S','SiContact'  ]
# ctrl_names = ['GateContact_P1','GateContact_P2','GateContact_S','SiContact']

ctrl_vals = pot.process_nextnano.get_ctrl_vals(potential) # Must make sure that ctrl_vals are sorted
ctrl_vals

# Now we define the field types we wish to write 2D slices for. Either potential or the electric field.
pot.process_nextnano.write_data(input_nextnano,output_preprocessed, slice=z, f_type=['potential','field'])

loaded_data_pot = pot.load_potentials(ctrl_vals, ctrl_names,
                                  f_type='pot', f_dir=pot_dir,
                                  f_dis_units='nm', f_pot_units='eV')

loaded_data_field = pot.load_potentials(ctrl_vals, ctrl_names,
                                  f_type='electric', f_dir=pot_dir,
                                  f_dis_units='nm', f_pot_units='V/nm')

potential_int = pot.build_interpolator(loaded_data_pot,
                                         constants=csts)
e_field_int = pot.build_interpolator(loaded_data_field,
                                         constants=csts)

# vg = [0.1,0.2,0.2,0.2325,0.1]
vg = [0.0,0.1,0.02,0.1,0.0, 0.0,0.0]
# vg = [0.4,0.6,0.0,0.0]

# potential_int.plot(vg, plot_type='1D', y_slice=0,)
# potential_int.plot(vg)
# e_field_int.plot(vg)


# plotting delta_g for N number of plunger gates and N+2 tunnel gates

# delta_g_vals = []
for idx, param in enumerate(loaded_data_pot['ctrl_vals']):
    
    # Left dot ------------------------------------------

    # restrict x-coordinates to left dot
    _2d_coords = loaded_data_pot['coords']

    x = _2d_coords.x
    x_adj = x[x<0]
    y = _2d_coords.y

    idx_l = len(x_adj)

    # pot_temp = loaded_data_pot['potentials'][idx][:,idx_l:] = 0.0
    pot_L = loaded_data_pot['potentials'][idx][:,:idx_l]

    # gparams = pot.GridParameters(x,y)
    gparams = pot.GridParameters(x_adj,y)

    # new_pot = -1*loaded_data_pot['potentials'][idx][:,:idx_l]
    new_pot = -1*pot_L
    gparams.update_potential(new_pot)
    
    
    
    _, e_vecs = qt.solvers.solve_schrodinger_eq(csts, gparams, n_sols=2)
    wf = e_vecs[:,:,0]
    # wf = e_vecs[:,:,1]

    dummy_e_int = lambda V: loaded_data_field['electric'][
                    np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]

    plt.figure()
    plt.imshow(np.real(np.multiply(wf,wf.conj())))
    plt.title(f'left wave function for {param}')

    plt.figure()
    plt.imshow(gparams.potential)
    # plt.plot(gparams.potential[64,:])
    plt.title(f'left e_field for {param}')

    # Right dot ------------------------------------------

    # restrict x-coordinates to left dot
    _2d_coords = loaded_data_pot['coords']

    x = _2d_coords.x
    x_adj = x[x>0]
    y = _2d_coords.y

    idx_r = len(x_adj)

    # pot_temp = loaded_data_pot['potentials'][idx][:,idx_l:] = 0.0
    pot_R = loaded_data_pot['potentials'][idx][:,idx_l:idx_l+idx_r]

    # gparams = pot.GridParameters(x,y)
    gparams = pot.GridParameters(x_adj,y)

    # new_pot = -1*loaded_data_pot['potentials'][idx][:,idx_l+1:idx_r]
    new_pot = -1*pot_R
    gparams.update_potential(new_pot)
    

    _, e_vecs = qt.solvers.solve_schrodinger_eq(csts, gparams, n_sols=2)
    wf = e_vecs[:,:,0]
    # wf = e_vecs[:,:,1]

    plt.figure()
    plt.imshow(np.real(np.multiply(wf,wf.conj())))
    plt.title(f'right wave function for {param}')

    plt.figure()
    # plt.imshow(gparams.potential)
    plt.plot(gparams.potential[64,:])
    plt.title(f'right e_field for {param}')

    if idx == 0:
        break

    # plt.figure()
    # plt.imshow(dummy_e_int(param)[60,:])
    # plt.title(f'1D e_field for {param}')


    # plt.figure()
    # plt.plot(np.real(wf[60,:]))
    # plt.title(f'1D wave function for {param}')

    # plt.show()


    # fig, ax = plt.subplots(3,2)

    # z = 0.45
    # # nearest_ind, nearest_slice = hp.find_nearest(loaded_data_field['coords']['z'], z)

    # # pdf
    # ax[0,0].imshow(np.real(np.multiply(wf,wf.conj())))
    # ax[0,0].set_title(f'$|\psi(x,y)|^2$')
    # # ax[0,0].set_xlabel('x (nm)')
    # # ax[0,0].set_ylabel('y (nm)')

    # # pdf
    # ax[1,0].imshow(np.real(wf))
    # ax[1,0].set_title(f'$\psi(x,y)$')
    # # ax[0,0].set_xlabel('x (nm)')
    # # ax[0,0].set_ylabel('y (nm)')
    
    # # nearest_ind, nearest_slice = hp.find_nearest(data_dict[iter]['coord']['x'], z)
    # # potential at some z-values
    # img = ax[2,1].imshow(dummy_e_int(param))
    # ax[2,1].set_title(f'$|E(x,y)|$')
    # # ax[0,1].set_xlabel('y (nm)')
    # # ax[0,1].set_ylabel('z (nm)')

    # y = 0
    # nearest_ind, nearest_slice = hp.find_nearest(loaded_data_field['coords'].y, y)
    # # pdf
    # ax[0,1].imshow(dummy_e_int(param)[nearest_ind,:])
    # ax[0,1].set_title(f'$|E(x)|$ @ y={nearest_slice}')
    # # ax[0,0].set_xlabel('x (nm)')
    # # ax[0,0].set_ylabel('y (nm)')
    
    # # nearest_ind, nearest_slice = hp.find_nearest(data_dict[iter]['coord']['x'], z)
    # # potential at some z-values
    # ax[1,1].imshow(np.real(wf[nearest_ind,:]))
    # ax[1,1].set_title(f'$\psi(x) @ y={nearest_slice}$')
    # # ax[0,1].set_xlabel('y (nm)')
    # # ax[0,1].set_ylabel('z (nm)')


    # x = 0
    # nearest_ind, nearest_slice = hp.find_nearest(loaded_data_field['coords'].x, x)
    # # nearest_ind, nearest_slice = hp.find_nearest(data_dict[iter]['coord']['x'], z)
    # # potential at some z-values
    # ax[2,1].imshow(np.real(wf[:,nearest_ind]))
    # ax[2,1].set_title(f'$\psi(y) @ x={nearest_slice}$')
    # # ax[0,1].set_xlabel('y (nm)')
    # # ax[0,1].set_ylabel('z (nm)')

    # fig.suptitle(f'data for {param}')
    # fig.colorbar(img)


    # fig, ax = plt.subplots(3,1)

    # z = 0.45
    # # nearest_ind, nearest_slice = hp.find_nearest(loaded_data_field['coords']['z'], z)

    # # pdf
    # ax[0,0].imshow(np.squeeze(np.real(np.multiply(wf,wf.conj()))))
    # ax[0,0].set_title(f'$|\psi(x,y)|^2$')
    # # ax[0,0].set_xlabel('x (nm)')
    # # ax[0,0].set_ylabel('y (nm)')

    # # pdf
    # ax[1,0].imshow(np.real(wf))
    # ax[1,0].set_title(f'$\psi(x,y)$')
    # # ax[0,0].set_xlabel('x (nm)')
    # # ax[0,0].set_ylabel('y (nm)')
    
    # # nearest_ind, nearest_slice = hp.find_nearest(data_dict[iter]['coord']['x'], z)
    # # potential at some z-values
    # img = ax[2,1].imshow(dummy_e_int(param))
    # ax[2,1].set_title(f'$|E(x,y)|$')
    # # ax[0,1].set_xlabel('y (nm)')
    # # ax[0,1].set_ylabel('z (nm)')


    # fig.suptitle(f'data for {param}')
    # fig.colorbar(img)




# plt.figure()
# # delta_g_vals = []
# for idx, param in enumerate(loaded_data_pot['ctrl_vals']):
#     new_pot = loaded_data_pot['potentials'][idx]
#     gparams.update_potential(new_pot)
#     _, e_vecs = qt.solvers.solve_schrodinger_eq(csts, gparams, n_sols=2)
#     wf = e_vecs[:,:,0]
#     stark_shift = ss.starkshift.StarkShift(gparams, csts)
#         #lambda to do the g-factor of discrete data rather than interpolated

#     dummy_e_int = lambda V: loaded_data_field['electric'][
#                     np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]

#     dg = stark_shift.delta_g(dummy_e_int, [param], ctrl_names,
#                                             wavefuncs=[wf])['delta_g_1'][0]

#     test = stark_shift.delta_g(dummy_e_int, [param], ctrl_names,
#                                             wavefuncs=[wf])
#     # test2 = stark_shift.delta_g(dummy_e_int, [param], ctrl_names)

#     if idx == 0:
#         plt.plot(param[0],dg,'.r', label='V1')
#         plt.plot(param[1],dg,'.g', label='V2')
#         plt.plot(param[2],dg,'.b', label='V3')
#         plt.plot(param[3],dg,'.k', label='V4')
#         plt.plot(param[4],dg,'.y', label='V5')
#     else:
#         plt.plot(param[0],dg,'.r')
#         plt.plot(param[1],dg,'.g')
#         plt.plot(param[2],dg,'.b')
#         plt.plot(param[3],dg,'.k')
#         plt.plot(param[4],dg,'.y')
#     plt.legend()

#     # delta_g_vals.append(dg)

plt.show()