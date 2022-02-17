import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import qudipy as qd
import qudipy.potential as pot
import qudipy.utils.helpers as hp

import numpy as np
import matplotlib.pyplot as plt

import string

# Define the source directory for the nextnano3D simulation data.
input_nextnano =    os.path.join(os.getcwd(), 'tutorials', 'QuDiPy tutorial data','Nextnano simulations test')
# Define a directory location which will contain the pre-processed potential data for a given 2D slice. If this directory doesn't exist currently then it will be created.
output_preprocessed = os.path.join(os.getcwd(),  'tutorials', 'QuDiPy tutorial data','Pre-processed potentials','Pre-processed_data')

print(input_nextnano)
print(output_preprocessed)
print('-----------------------------')

# Import all of the neccessary potential data and meta data for the simulation runs.
# Potential is a dictionary data type containing nested list.
# Each element of the dictionary contains [list of control valuess, list of potential data, dictionary for x,y, and z cordinate data].
potential = pot.process_nextnano.import_dir(input_nextnano, nn_vars=['v1','v2','v3','v4','v5'], show_files=True)

# print(input_nextnano)

# # Note the nested dictionary key structure
# print(potential.keys())
# print('-----------------------------=========================')
# print(potential[0].keys())
# print('-----------------------------')
# print(potential[0]['ctrl_names'])
# print('-----------------------------')
# print(potential[0]['ctrl_values'])
# print('-----------------------------++++++++++++++++++++++++')
# print(potential[0]['coord'].keys())
# print('-----------------------------')
# print(potential[1]['ctrl_names'])
# print('-----------------------------')
# print(potential[1]['ctrl_values'])
# print('-----------------------------++++++++++++++++++++++++')
# print(potential[1]['coord'].keys())






