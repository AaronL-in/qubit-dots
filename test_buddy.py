import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import qudipy as qd
import qudipy.potential as pot
import qudipy.utils.helpers as hp

import numpy as np
import matplotlib.pyplot as plt

# Define the source directory for the nextnano3D simulation data.
input_nextnano =    os.path.join(os.getcwd(), 'tutorials', 'QuDiPy tutorial data','Nextnano simulations')
# Define a directory location which will contain the pre-processed potential data for a given 2D slice. If this directory doesn't exist currently then it will be created.
output_preprocessed = os.path.join(os.getcwd(),  'tutorials', 'QuDiPy tutorial data','Pre-processed potentials','Pre-processed_data')

print(input_nextnano)
print(output_preprocessed)
print('-----------------------------')

# Import all of the neccessary potential data and meta data for the simulation runs.
# Potential is a dictionary data type containing nested list.
# Each element of the dictionary contains [list of control valuess, list of potential data, dictionary for x,y, and z cordinate data].
potential = pot.process_nextnano.import_folder(input_nextnano, file_import_display=True)

# Note the nested dictionary key structure
print(potential.keys())
print(potential[0].keys())
print(potential[0]['coord'].keys())

# Enter the desired z coordinate to take a cross-section of the x-y plane. 
# The  slice along the z axis will be preformed for the nearest simulated z-coordinate to the user defined coordinate.
z = 0.45

# Now we define the field types we wish to write 2D slices for. Either potential or the electric field.
pot.process_nextnano.write_data(input_nextnano,output_preprocessed, slice=z, f_type=['potential','field'])


_, nearest_slice = hp.find_nearest(potential[0]['coord']['z'], z)

pot_dir = output_preprocessed + '_for_nearest_slice{:.3e}'.format(nearest_slice)

# Specify the control voltage names (C#NAME as mentioned above)
ctrl_names = ['V1','V2','V3','V4','V5']

# Specify the control voltage values you wish to load.
# The cartesian product of all these supplied voltages will be loaded and MUST exist in the directory.
V1 = [0.1]
V2 = [0.2]
V3 = [0.2]
V4 = [0.2, 0.22, 0.24, 0.26]
V5 = [0.1]

ctrl_vals = [V1, V2, V3, V4, V5] 

# Or extract voltage values from the the imported nextnano data  
ctrl_vals = pot.process_nextnano.retrieve_ctrl_vals(potential)


loaded_data = pot.load_potentials(ctrl_vals, ctrl_names,
                                  f_type='pot', f_dir=pot_dir,
                                  f_dis_units='nm', f_pot_units='eV')
# Additional keyword arguments are trim_x and trim_y which will trim the loaded potentials 
# to be only within the bounds specified by trim_x and trim_y

# We can check the fields of the dictionary returned to see what is stored.
loaded_data.keys()





