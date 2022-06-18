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

data_exist = False
plot_flag = False

# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','TMPLATE_5Gate_1.358E15_noRGrind')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','2gate_t1_d_sep_10')
# input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_dotsep_60nm')
input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_dotsep_60nm copy')
input_nextnano = os.path.join(sys.path[0], 'tutorials', 'QuDiPy tutorial data','Nextnano simulations','DQD_all')

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

    # ctrl_vals = pot.get_ctrl_vals(loaded_data_pot['ctrl_vals'])

    # print(np.shape(loaded_data_pot['ctrl_vals']))
    # print(np.shape(loaded_data_pot['ctrl_vals'])[0])


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


    shp = [len(ctrl_vals[1]), len(ctrl_vals[2]), len(ctrl_vals[3])]
    print(np.shape(np.array(ctrl_vals)))
    
    tup = ()
    for i in ctrl_vals:
        tup += (len(i),)

    gl_data = np.empty(tup)
    # gl_data[:] = np.NaN

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

        # assign to data array
        # ctrl_vals
        idx = []
        for n,i in enumerate(param):
            idice = np.where(ctrl_vals[n] == i)
            a = idice[0]
            aa = idice[0][0]

            idx.append([int(idice[0][0])])

        idex = np.ix_(*idx)
        
        gl_data[idex] = dgr

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

        # # load all gate voltages
        # all_gates = np.zeros((len(param),len(loaded_data_pot['ctrl_vals'])))
        # for i in range(len(param)):
        #     all_gates[i,idx] = param[i]

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

        # --------------------------------------------------------------------------
        # Calculate exchange

    data = {}
    data['gl'] = delta_gl_vals
    data['gr'] = delta_gr_vals
    data['pl'] = pl
    data['pr'] = pr
    data['t'] = t
    data['idx'] = idx_map
    data['ctrl_vals'] = loaded_data_pot['ctrl_vals']

    # create a binary pickle file
    # f = open("data2.pkl","wb")
    f = open("data3.pkl","wb")
    # write the python object (dict) to pickle file
    pk.dump(data,f)
    f.close()
    
    f = open("ctrl_vals.pkl","wb")
    # write the python object (dict) to pickle file
    pk.dump(ctrl_vals,f)
    f.close()

    f = open("gl_data.pkl","wb")
    # write the python object (dict) to pickle file
    pk.dump(gl_data,f)
    f.close()




else:
    # file_to_read = open("data2.pkl", "rb")
    file_to_read = open("data3.pkl", "rb")
    data = pk.load(file_to_read)

    file_to_read = open("ctrl_vals.pkl", "rb")
    ctrl_vals = pk.load(file_to_read)
    
    file_to_read = open("gl_data.pkl", "rb")
    gl_data = pk.load(file_to_read)







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
# test_data = np.array([data['gl'], data['pl'][::-1], data['pr'][::-1], data['t'][::-1]])
test_data = np.array([data['gl'], data['pl'], data['pr'], data['t']])

# g1 raw
# fig, (ax1,ax2,ax3) = plt.subplots(3,1)
fig, (ax2,ax3) = plt.subplots(2,1)

# g1(v1)
# ax1.plot(v1, g3_arr[0], '-o')
# ax1.set_title("g1")
# ax1.set_xlabel("v1")
# ax1.set_ylabel("g")

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

    x = w1_group[idx+1]
    y = g1_group[idx+1]

    f = interp1d(x,y)
    x_int = np.linspace(min(x),max(x), 100)
    ax2.plot(x_int, f(x_int), '-k', label='Interpolation')

    ax2.plot(w1_group[idx+1],g1_group[idx+1], 'o', label=f'W1 = {i}')
    
    
ax2.set_title("g1")
ax2.set_xlabel("v2")
ax2.set_ylabel("g")

lines, labels = ax2.get_legend_handles_labels()
ax2.legend(labels, bbox_to_anchor=(1.2, 1))

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
    x = v2_group[idx+1]
    y = g1_group[idx+1]

    f = interp1d(x,y)
    x_int = np.linspace(min(x),max(x), 100)
    ax3.plot(x_int, f(x_int), '-k', label='Interpolation')

    ax3.plot(x, y, 'o', label=f'V2 = {i}')

ax3.set_title("g1")
ax3.set_xlabel("t")
ax3.set_ylabel("g")

lines, labels = ax3.get_legend_handles_labels()
ax3.legend(labels, bbox_to_anchor=(1.2, 1))

plt.tight_layout()

save_path = os.path.join(png_dir, 'g1_1d_dots_array.png')
plt.savefig(save_path, dpi=1000)

# gr ---------------------------------------------------

# test_data = np.array([data['gr'], data['pl'][::-1], data['pr'][::-1], data['t'][::-1]])
test_data = np.array([data['gr'], data['pl'], data['pr'], data['t']])

# g2 raw
# fig, (ax1,ax2,ax3) = plt.subplots(3,1)
fig, (ax2,ax3) = plt.subplots(2,1)

# g2(v1)
# ax1.plot(v1, g3_arr[0], '-o')
# ax1.set_title("g2")
# ax1.set_xlabel("v1")
# ax1.set_ylabel("g")

# g2(v2)

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

    x = w1_group[idx+1]
    y = g1_group[idx+1]

    f = interp1d(x,y)
    x_int = np.linspace(min(x),max(x), 100)
    ax2.plot(x_int, f(x_int), '-k', label='Interpolation')

    ax2.plot(w1_group[idx+1],g1_group[idx+1], 'o', label=f'W1 = {i}')
    
ax2.set_title("g2")
ax2.set_xlabel("v2")
ax2.set_ylabel("g")

lines, labels = ax2.get_legend_handles_labels()
ax2.legend(labels, bbox_to_anchor=(1.2, 1))

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
    x = v2_group[idx+1]
    y = g1_group[idx+1]


    f = interp1d(x,y)
    x_int = np.linspace(min(x),max(x), 100)
    ax3.plot(x_int, f(x_int), '-k', label='Interpolation')

    ax3.plot(x, y, 'o', label=f'V2 = {i}')


ax3.set_title("g2")
ax3.set_xlabel("t")
ax3.set_ylabel("g")

lines, labels = ax3.get_legend_handles_labels()
ax3.legend(labels, bbox_to_anchor=(1.2, 1))

plt.tight_layout()

save_path = os.path.join(png_dir, 'g2_1d_dots_array.png')
plt.savefig(save_path, dpi=1000)

# 3D -------------------------------------------------------------------
v1 = np.array(data['pl'])
v2 = np.array(data['pr'])
w1 = np.array(data['t'])

COR = []
for i in range(len(ctrl_vals)):
    COR.append(np.linspace(min(ctrl_vals[i]), max(ctrl_vals[i]), len(ctrl_vals[i])))

M = np.meshgrid(v2, w1)

# g1 interpolation -----------------------------
# cartcoord = list(zip(v2,w1))
# V2 = np.linspace(min(v2), max(v2), 100)
# W1 = np.linspace(min(w1), max(w1), 100)
# V2, W1 = np.meshgrid(v2, w1)
interp = InterpolateND(gl_data, *ctrl_vals)
# interp = pot.build_interpolator(data['gl'], constants=csts)
# G = interp(V2,W1)#*1E4
interp(*COR)#*1E4

interp.Crossection2D(phys_param='gl')
interp.Crossection3D(COR, phys_param='gl')

plt.show()





