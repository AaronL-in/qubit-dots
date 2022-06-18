from configparser import Interpolation
import os, sys

from imageio import save
sys.path.append(os.path.dirname(os.getcwd()))


import qudipy.potential as pot

import qudipy.potential as pot
import qudipy.qutils as qt
import qudipy.utils.helpers as hp

from qudipy.potential import InterpolateND

import qudipy.starkshift as ss
from qudipy.utils.constants import Constants
import qudipy.potential.InterpolateND as interp_nd

from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import LinearNDInterpolator


import numpy as np
import matplotlib.pyplot as plt
import pickle as pk
from tqdm import tqdm

csts = Constants('Si/SiO2')

data_exist = True
plot_flag = False

# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','TMPLATE_5Gate_1.358E15_noRGrind')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','2gate_t1_d_sep_10')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_dotsep_60nm')
input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_dotsep_60nm copy')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_all')

output_preprocessed = os.path.join(sys.path[0],'tutorials', 'QuDiPy tutorial data','Pre-processed potentials','Pre-processed_data')

png_dir = os.path.join(sys.path[0],'data')

if not data_exist:
        
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
    # ctrl_vals

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

    # # vg = [0.1,0.2,0.2,0.2325,0.1]
    # vg = [0.0,0.1,0.02,0.1,0.0, 0.0,0.0]
    # vg = [0.1,0.02,0.1]
    # # vg = [0.4,0.6,0.0,0.0]

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
        ele_L = -1*np.gradient(pot_L)[-1]
        pot_R = loaded_data_pot['potentials'][idx].copy()
        ele_R = -1*np.gradient(pot_R)

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

        # Left dot ------------------------------------------
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

        # append delta_g factors for given gate voltages
        delta_gl_vals.append(dgl)
        delta_gr_vals.append(dgr)
        pl.append(param[1])
        pr.append(param[2])
        t.append(param[3])
        idx_map.append(idx)

        if plot_flag == True:
            plt.figure()
            plt.imshow(pot_T)
            plt.title(f'$V_L = {param[1]}, V_R = {param[2]}, W = {param[3]}$')
            plt.xlabel('x')
            plt.ylabel('y')
            save_path = os.path.join(png_dir, f'total potential for {param}.png')
            plt.savefig(save_path, dpi=1000)
            plt.colorbar()

            plt.figure()
            plt.imshow(dummy_e_intl([param]))
            plt.title(f'$V_L = {param[1]}, V_R = {param[2]}, W = {param[3]}$')
            plt.xlabel('x')
            plt.ylabel('y')
            
            save_path = os.path.join(png_dir, f'total electric field for {param}.png')
            plt.savefig(save_path, dpi=1000)
            plt.colorbar()


            # fig, ax = plt.subplots(2,2)
            # img = ax[0,0].imshow(np.real(np.multiply(wfr,wfr.conj())))
            # ax[0,0].set_title(f'$|\psi_R(x,y)|^2$')
            # ax[0,0].set_xlabel('x')
            # ax[0,0].set_xlabel('y')

            # img = ax[1,0].imshow(gparamsr.potential)
            # # plt.plot(gparams.potential[64,:])
            # ax[1,0].set_title(f'$V_R(x,y)$')
            # ax[1,0].set_xlabel('x')
            # ax[1,0].set_xlabel('y')

            # img = ax[0,1].imshow(np.real(np.multiply(wfl,wfl.conj())))
            # ax[0,1].set_title(f'$|\psi_L(x,y)|^2$')
            # ax[0,1].set_xlabel('x')
            # ax[0,1].set_xlabel('y')

            # ax[1,1].imshow(gparamsl.potential)
            # # plt.plot(gparams.potential[64,:])
            # ax[1,1].set_title(f'$V_L(x,y)$')
            # ax[1,1].set_xlabel('x')
            # ax[1,1].set_xlabel('y')

            # plt.show()
            # save_path = os.path.join(png_dir, f'potential slice for {param}.png')
            # plt.savefig(save_path, dpi=1000)

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

    data = {}
    data['gl'] = delta_gl_vals
    data['gr'] = delta_gr_vals
    data['pl'] = pl
    data['pr'] = pr
    data['t'] = t
    data['idx'] = idx_map
    data['ctrl_vals'] = loaded_data_pot['ctrl_vals']

    # np.savetxt('data.txt', data)

    # create a binary pickle file
    f = open("data.pkl","wb")
    # write the python object (dict) to pickle file
    pk.dump(data,f)
    f.close()

else:
    file_to_read = open("data.pkl", "rb")
    data = pk.load(file_to_read)


V1, V2, W1 = np.meshgrid(data['pl'],data['pr'],data['t'])

g1_arr = np.array([data['gl'], data['pl']])
g1_sort = np.sort(g1_arr, axis=1)
v1 = g1_sort[1]

g2_arr = np.array([data['gl'], data['pr']])
g2_sort = np.sort(g2_arr, axis=1)
v2 = g2_sort[1]

g3_arr = np.array([data['gl'], data['t']])
g3_sort = np.sort(g3_arr, axis=1)
w1 = g3_sort[1]

# raw data  -----------------------------------------------------------------
test_data = np.array([data['gl'], data['pl'], data['pr'], data['t']])

# g1 raw
fig, (ax1,ax2,ax3) = plt.subplots(3,1)

# g1(v1)
ax1.plot(v1, g3_arr[0], '-o')
ax1.set_title("g1")
ax1.set_xlabel("v1")
ax1.set_ylabel("g")

# g1(v2)

# sort row of interest
sort_group = 2
xdata = 3
ydata = 0

idx = test_data[sort_group,:].argsort()
sort_data = test_data[:,idx]

# transpose data to sort  multiple columns
transpose_data = sort_data.T

key1 = transpose_data[:,sort_group]
key2 = transpose_data[:,xdata]

var = np.lexsort((key2, key1))
final_data = transpose_data[var].T

# group data based on soreted elements
split_element = np.unique(final_data[sort_group,:], return_index=True)
g1_group = np.split(final_data[ydata,:], split_element[1])
w1_group = np.split(final_data[xdata,:], split_element[1])

for idx, i in enumerate(split_element[0]):
    ax2.plot(w1_group[idx+1],g1_group[idx+1], '-o', label=f'W1 = {i}')
    
ax2.set_title("g1")
ax2.set_xlabel("v2")
ax2.set_ylabel("g")
ax2.legend()

# g1(t)

# sort row of interest
sort_group = 3
xdata = 2
ydata = 0

idx = test_data[sort_group,:].argsort()
sort_data = test_data[:,idx]

# transpose data to sort  multiple columns
transpose_data = sort_data.T

key1 = transpose_data[:,sort_group]
key2 = transpose_data[:,xdata]

var = np.lexsort((key2, key1))
final_data = transpose_data[var].T

# group data based on soreted elements
split_element = np.unique(final_data[sort_group,:], return_index=True)
g1_group = np.split(final_data[ydata,:], split_element[1])
v2_group = np.split(final_data[xdata,:], split_element[1])

for idx, i in enumerate(split_element[0]):
    ax3.plot(v2_group[idx+1],g1_group[idx+1], '-o', label=f'V2 = {i}')
ax3.set_title("g1")
ax3.set_xlabel("t")
ax3.set_ylabel("g")
ax3.legend()

plt.tight_layout()

save_path = os.path.join(png_dir, 'g1_1d_dots_array.png')
plt.savefig(save_path, dpi=1000)

# 3D -------------------------------------------------------------------

v1 = np.array(data['pl'])
v2 = np.array(data['pr'])
w1 = np.array(data['t'])


# g1 interpolation -----------------------------

# coord = []
# coord.append(v1)
# coord.append(v2)
# coord.append(w1)

cartcoord = list(zip(v2,w1))
V2 = np.linspace(min(v2), max(v2), 100)
W1 = np.linspace(min(w1), max(w1), 100)
# V2, W1 = np.meshgrid(v2, w1)
interp = InterpolateND(data['gl'], v1, v2, w1)
# interp = pot.build_interpolator(data['gl'], constants=csts)
# G = interp(V2,W1)#*1E4
interp(V2,W1)#*1E4

interp.Crossection2D(phys_param='gl')

# G[G == 0.0] = 'nan'

# # fig = plt.figure(figsize=(16, 12))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # img = ax.imshow(G)
# img = ax.scatter(V2,W1,G)
# cb = plt.colorbar(img, pad=0.2)
# ax.set_title("g1")
# ax.set_xlabel("v2")
# ax.set_ylabel("t")
# save_path = os.path.join(png_dir, 'g1_3d.png')
# plt.savefig(save_path, dpi=1000)

# cartcoord = list(zip(v2,w1))
# V2 = np.linspace(min(v2), max(v2), 100)
# W1 = np.linspace(min(w1), max(w1), 100)
# # V2, W1 = np.meshgrid(v2, w1)
# interp = LinearNDInterpolator(cartcoord, data['gl'], fill_value=0)
# # interp = pot.build_interpolator(data['gl'], constants=csts)
# G = interp(V2,W1)#*1E4
# G[G == 0.0] = 'nan'

# # fig = plt.figure(figsize=(16, 12))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # img = ax.imshow(G)
# img = ax.scatter(V2,W1,G)
# cb = plt.colorbar(img, pad=0.2)
# ax.set_title("g1")
# ax.set_xlabel("v2")
# ax.set_ylabel("t")
# save_path = os.path.join(png_dir, 'g1_3d.png')
# plt.savefig(save_path, dpi=1000)

# cartcoord = list(zip(v2,w1))
# # Y = np.linspace(min(v2), max(v2))
# # Z = np.linspace(min(w1), max(w1))
# V2, W1 = np.meshgrid(v2, w1)
# interp = LinearNDInterpolator(cartcoord, data['gl'], fill_value=0)
# G = interp(V2,W1)#*1E4
# G[G == 0.0] = 'nan'

# # fig = plt.figure(figsize=(16, 12))
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # img = ax.imshow(G)
# img = ax.scatter(V2,W1,G)
# cb = plt.colorbar(img, pad=0.2)
# ax.set_title("g1")
# ax.set_xlabel("v2")
# ax.set_ylabel("t")
# save_path = os.path.join(png_dir, 'g1_3d.png')
# plt.savefig(save_path, dpi=1000)



plt.show()





