import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import qudipy as qd
import qudipy.potential as pot
import qudipy.utils.helpers as hp

import numpy as np
import matplotlib.pyplot as plt

import string

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


test = ['# AVS/Express field file\n', 
        '#\n', 
        'ndim = 3\n',
        'dim1 = 103\n',
        'dim2 = 35\n',
        'dim3 = 78\n',
        'nspace = 3\n', 
        'veclen = 1\n', 
        'data = double\n', 
        'field = rectilinear\n', 
        'label = potential\n', 
        '\n', 
        'variable 1 file=potential.dat filetype=ascii skip=0 offset=0 stride=1\n', 
        'coord 1 file=potential.coord filetype=ascii skip=0 offset=0 stride=1\n', 
        'coord 2 file=potential.coord filetype=ascii skip=104 offset=0 stride=1\n', 
        'coord 3 file=potential.coord filetype=ascii skip=140 offset=0 stride=1\n']

for t in test:
    print(t.strip(string.punctuation).strip(string.ascii_letters))

print(test[3].split())

print([int(i) for i in test[3].split() if i.isdigit()])
print([int(i) for i in test[3] if i.isdigit()])




