from cProfile import label
from configparser import Interpolation
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))


import qudipy.potential as pot

import qudipy.potential as pot
import qudipy.qutils as qt
import qudipy.utils.helpers as hp

import qudipy.starkshift as ss
from qudipy.utils.constants import Constants

from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator


import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

csts = Constants('Si/SiO2')

# pot_dir = os.path.join(sys.path[0], 'QuDiPy tutorial data','Nextnano simulations','TMPLATE_5Gate_1.358E15_noRGrind')

input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','TMPLATE_5Gate_1.358E15_noRGrind')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','2gate_t1_d_sep_10')
input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_dotsep_60nm')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_all')

output_preprocessed = os.path.join(sys.path[0],'tutorials', 'QuDiPy tutorial data','Pre-processed potentials','Pre-processed_data')

#importing potential
potential = pot.process_nextnano.import_dir(input_nextnano, show_files=True)


# Desired Slice
z = -0.2
_, nearest_slice = hp.find_nearest(potential[0]['coord']['z'], z)

print(f'The nearest slice for z = {z} is: {nearest_slice}')
#specifying control values and names
pot_dir = output_preprocessed + '_for_nearest_slice{:.3e}'.format(nearest_slice) + '/'

# TODO: allow user to extract ctrl_values for only certain ctrl_names
ctrl_names = ['V1','V2','V3','V4','V5']
# ctrl_names = ['V_P','VTUN','BIAS']
# ctrl_names = ['V_P','VTUN','BIAS']
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

# potential_int = pot.build_interpolator(loaded_data_pot,
#                                          constants=csts)
# e_field_int = pot.build_interpolator(loaded_data_field,
                                        #  constants=csts)

# vg = [0.1,0.2,0.2,0.2325,0.1]
vg = [0.0,0.1,0.02,0.1,0.0, 0.0,0.0]
# vg = [0.4,0.6,0.0,0.0]

# potential_int.plot(vg, plot_type='1D', y_slice=0,)
# potential_int.plot(vg)
# e_field_int.plot(vg)


delta_gl_vals = []
delta_gr_vals = []
ex = []
pl = []
pr = []
t = []
idx_map = []
# for idx, param in enumerate(tqdm(loaded_data_pot['ctrl_vals'])):
for idx, param in enumerate(tqdm(loaded_data_pot['ctrl_vals'])):

    print(f'Processing ')
    # --------------------------------------------------------------------------
    # plotting delta_g for N number of plunger gates and N+2 tunnel gates
    pot_T = loaded_data_pot['potentials'][idx].copy()
    pot_L = loaded_data_pot['potentials'][idx].copy()
    ele_L = -1*np.gradient(pot_L)
    pot_R = loaded_data_pot['potentials'][idx].copy()
    ele_R = -1*np.gradient(pot_L)

    _2d_coords = loaded_data_pot['coords']
    x = _2d_coords.x
    y = _2d_coords.y

    gparams = pot.GridParameters(x,y)
    gparamsl = pot.GridParameters(x,y)
    gparamsr = pot.GridParameters(x,y)

    # Right dot ------------------------------------------

    # restrict x-coordinates to left dot
    x_r = x[x>0]
    idx_r = len(x_r)
    n,m = np.shape(pot_R)
    base = np.amin(pot_R)
    pot_R[:,:idx_r] = base

    gparamsr.update_potential(-1*pot_R)
    
    _, e_vecsr = qt.solvers.solve_schrodinger_eq(csts, gparamsr, n_sols=2)
    wfr = e_vecsr[:,:,0]
    # wf = e_vecs[:,:,1]


    # g-factor
    stark_shiftr = ss.starkshift.StarkShift(gparamsr, csts)
            #lambda to do the g-factor of discrete data rather than interpolated

    dummy_e_intr = lambda V: loaded_data_field['electric'][np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]

    # dummy_e_intr = lambda V: ele_L[np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]

    dgr = stark_shiftr.delta_g(dummy_e_intr, [param], ctrl_names,
                                                wavefuncs=[wfr])['delta_g_1'][0]

    # Left dot ------------------------------------------
    x_l = x[x<0]
    idx_l = len(x_l)
    n,m = np.shape(pot_L)
    base = np.amin(pot_L)
    pot_L[:,idx_l:] = base

    gparamsl.update_potential(-1*pot_L)
    
    
    _, e_vecsl = qt.solvers.solve_schrodinger_eq(csts, gparamsl, n_sols=2)
    wfl = e_vecsl[:,:,0]


    # g-factor
    stark_shiftl = ss.starkshift.StarkShift(gparamsl, csts)
            #lambda to do the g-factor of discrete data rather than interpolated

    # dummy_e_intl = lambda V: ele_R[np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]
    dummy_e_intl = lambda V: loaded_data_field['electric'][np.nonzero(np.array(ctrl_vals[0]) == V[0])[0][0]]

    dgl = stark_shiftl.delta_g(dummy_e_intl, [param], ctrl_names,
                                                wavefuncs=[wfl])['delta_g_1'][0]

    # append delta_g factors for given gate voltages
    delta_gl_vals.append(dgl)
    delta_gr_vals.append(dgr)
    pl.append(param[1])
    pr.append(param[2])
    t.append(param[3])
    idx_map.append(idx)

    # fig, ax = plt.subplots(3,2)
    # img = ax[0,0].imshow(np.real(np.multiply(wfr,wfr.conj())))
    # ax[0,0].set_title(f'right wave function')

    # img = ax[1,0].imshow(gparamsr.potential)
    # # plt.plot(gparams.potential[64,:])
    # ax[1,0].set_title(f'right potential')

    # img = ax[0,1].imshow(np.real(np.multiply(wfl,wfl.conj())))
    # ax[0,1].set_title(f'left wave function')

    # ax[1,1].imshow(gparamsl.potential)
    # # plt.plot(gparams.potential[64,:])
    # ax[1,1].set_title(f'left potential')

    # img = ax[2,0].imshow(-1*pot_T)
    # # ax[2,0].plot(-1*pot_T[64,:])
    # ax[2,0].set_title(f'potential')

    # ax[2,1].imshow(-1*pot_T)
    # # ax[2,1].plot(-1*pot_T[64,:])
    # ax[2,1].set_title(f'potential')
    # fig.suptitle(f'{param}')
    # # fig.colorbar(img)
    # fig.savefig(f'potential for {param}.png')

    # --------------------------------------------------------------------------
    # Calculate exchange

    # gparams.update_potential(pot_T)

    # ne = 1
    # n_xy_ho = (4,4)
    # n_se_orbs = 2
    # ommega_guess = omega_guess = 0.025 * csts.hbar / (csts.me * csts.a_B**2)
    # data_dir = ''

    # ens, __ = qt.solvers.solve_many_elec_SE(gparams, n_elec=ne,
    #             n_xy_ho=n_xy_ho,
    #             n_se=n_se_orbs,
    #             n_sols=4,
    #             consts=csts,
    #             optimize_omega=False,
    #             omega=omega_guess,
    #             cme_dir= data_dir,
    #             spin_subspace=[0])
    # J = ens[1] - ens[0]
    # print('exchange value is: ', J,
    # '\nenergies are: ', ens)
    # ex.append(J)


plot_array = np.array(delta_gl_vals)

print(plot_array)
print('----------------------------')
plot_array = np.vstack([plot_array, delta_gr_vals])
print(plot_array)
print('----------------------------')
plot_array = np.vstack([plot_array, pl])
print(plot_array)
print('----------------------------')
plot_array = np.vstack([plot_array, pr])
print(plot_array)
print('----------------------------')
plot_array = np.vstack([plot_array, t])
print(plot_array)
print('----------------------------')
plot_array = np.vstack([plot_array, idx_map])

print(f' not sorted: {plot_array}')


# V1, V2, W1, g1 = np.meshgrid(plot_array[2,:], plot_array[3,:], plot_array[4,:], plot_array[0,:])
V1, V2, W1 = np.meshgrid(pl,pr,t)

v1 = np.array(pl)
v2 = np.array(pr)
w1 = np.array(t)

# fig, ax = plt.subplots(3,1)
plt.figure()
# v1/v2
cartcoord = list(zip(v1,v2))
X = np.linspace(min(v1), max(v1))
Y = np.linspace(min(v2), max(v2))
X, Y = np.meshgrid(X, Y)
interp = LinearNDInterpolator(cartcoord, delta_gl_vals, fill_value=0)
Z0 = interp(X, Y)/1.6E-19*1E4
img = plt.pcolormesh(X, Y, Z0)

# fig, ax = plt.subplots(3,1)
# # v1/v2
# cartcoord = list(zip(v1,v2))
# X = np.linspace(min(v1), max(v1))
# Y = np.linspace(min(v2), max(v2))
# X, Y = np.meshgrid(X, Y)
# interp = LinearNDInterpolator(cartcoord, delta_gl_vals, fill_value=0)
# Z0 = interp(X, Y)
# ax[0,0].pcolormesh(X, Y, Z0)
# ax[0,0].plot_title('$v_1/v_2$')
# # ax[1,0].plot_x_label('$v_1$')
# # ax[1,0].plot_y_label('$v_2$')

# # v1/t
# cartcoord = list(zip(v1,t))
# X = np.linspace(min(v1), max(v1))
# Y = np.linspace(min(t), max(t))
# X, Y = np.meshgrid(X, Y)
# interp = LinearNDInterpolator(cartcoord, delta_gl_vals, fill_value=0)
# Z0 = interp(X, Y)
# ax[1,0].pcolormesh(X, Y, Z0)
# ax[1,0].plot_title('$v_1/t$')
# # ax[1,0].plot_x_label('$v_1$')
# # ax[1,0].plot_y_label('$t$')

# v2/t
# cartcoord = list(zip(v1,v2))
# X = np.linspace(min(v2), max(v2))
# Y = np.linspace(min(t), max(t))
# X, Y = np.meshgrid(X, Y)
# interp = LinearNDInterpolator(cartcoord, delta_gl_vals, fill_value=0)
# Z0 = interp(X, Y)
# img = ax[2,0].pcolormesh(X, Y, Z0)
# ax[0,0].plot_title('$v_2/t$')
# # ax[1,0].plot_x_label('$v_2$')
# # ax[1,0].plot_y_label('$t$')


plt.colorbar(img) # Color Bar
plt.show()

# g1_int = RegularGridInterpolator(
#     points=[V1, V2, W1],
#      values=g1)

# G1 = g1_int(V1, V2, W1)

# v1 = np.linspace(-1,1,100)

# plt.figure()
# # plt.plot(v1, g1_int(v1), label='$g_1(v_1)')
# plt.pcolormesh(V1, V2, G1, shading='auto')
# plt.legend()

plt.figure()
# gl
sort_data = plot_array[:, np.argsort(plot_array[2, :])]
plt.plot(sort_data[2,:], sort_data[0,:], label='$g_l(P_l)$')
sort_data = plot_array[:, np.argsort(plot_array[3, :])]
plt.plot(sort_data[3,:], sort_data[0,:], label='$g_l(P_r)$')
sort_data = plot_array[:, np.argsort(plot_array[4, :])]
plt.plot(sort_data[4,:], sort_data[0,:], label='$g_l(t)$')

# gr
sort_data = plot_array[:, np.argsort(plot_array[2, :])]
plt.plot(sort_data[2,:], sort_data[1,:], label='$g_r(P_l)$')
sort_data = plot_array[:, np.argsort(plot_array[3, :])]
plt.plot(sort_data[3,:], sort_data[1,:], label='$g_r(P_r)$')
sort_data = plot_array[:, np.argsort(plot_array[4, :])]
plt.plot(sort_data[4,:], sort_data[1,:], label='$g_r(t)$')
plt.legend()
plt.savefig('sorted by ctrl_values.png')

print(f' sorted: {sort_data}')

print(f' delta gl: {delta_gl_vals}')
print(f' delta gr: {delta_gr_vals}')
print(f' pl: {pl}')
print(f' pr: {pr}')
print(f' t: {t}')
print(f' idx_map: {idx_map}')

# plt.figure()
# # gl
# plt.plot(pl,delta_gl_vals, label='$g_l(P_l)$')
# plt.plot(pr,delta_gl_vals, label='$g_l(P_r)$')
# plt.plot(t,delta_gl_vals, label='$g_l(P_t)$')
# # gr
# plt.plot(pl,delta_gr_vals, label='$g_r(P_l)$')
# plt.plot(pr,delta_gr_vals, label='$g_r(P_r)$')
# plt.plot(t,delta_gr_vals, label='$g_r(P_t)$')

# plt.legend()
plt.show()





