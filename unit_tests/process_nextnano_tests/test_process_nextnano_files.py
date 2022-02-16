'''
Unittests for the module potential.process_nextnano.py.

@author: Zach
'''

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pytest

import qudipy.potential as pot

'''
Instructions for running unittests:

1) Open cmd or anaconda promt and change directory to the unittest directory
2) Make sure tutorial data is downloaded and located in tutorial directory.
3) Run the unittest via unittest/pytest package:
    - Unittest is built into python; pytest must be downloaded with 
    "python -m pip install pytest".
    - Enter comand: "python -m unittest <filename.py>" or
        "python -m pytest <filename.py>"

Note: Markers are used to call unitest functions for specific aspects of the 
    test files. Marker declirations are done before the function.

    Example:

        First: Define custom marker in pytest.ini configuration file.

        Second: Add marker to desired function as follows.
        
            @pytest.mark.<marker name>
            def <function name>():
'''
'''
Define custom markers as follows:

markers =
        <marker 1 name>: <optional description>,
        <marker 2 name>,
        <marker 2 name>
'''

# Test data is being imported correctly
@pytest.mark.process_data
def test_data_importation(ctrl_data_input_path,ctrl_data):

    # Check that any data from path is imported (list is False if empty)
    # assert ctrl_data is False

    assert ctrl_data

    # Make sure every trial run was imported

    assert len(ctrl_data) == 5

    # Make sure a trial run was imported correctly

    # first sublist: control names
    assert len(ctrl_data[0]['ctrl_names']) == 5

    # second sublsit: potential list containing 3D information
    assert len(ctrl_data[0]['ctrl_values']) == 281190

    # third sublist: cordinates for potential list
    assert len(ctrl_data[0]['coord']) == 3

    # Make sure the structure of the data is correct
    assert type(ctrl_data[0]['ctrl_names']) is list
    assert type(ctrl_data[0]['ctrl_values']) is np.ndarray
    assert type(ctrl_data[0]['coord']) is dict

# parse_ctrl_items works and retrieve_ctrl_vals works
def test_ctrl_val_minipulation(ctrl_data_input_path, ctrl_data):
    
    # Make sure control names are parsed correctly
    for subdir, _, _ in os.walk(ctrl_data_input_path):

        # parse voltage information for the directory one level higher than
        # /output
        if subdir != ctrl_data_input_path and subdir[-6:] == 'output':
            gates = pot.parse_ctrl_items(subdir,'name')
            break

    assert gates == ['V1','V2','V3','V4','V5']
    
    # Make sure control values are combined correctly
    ctrl_vals = pot.process_nextnano.retrieve_ctrl_vals(ctrl_data)

    print(ctrl_data.keys)

    print(ctrl_vals)

    # assert ctrl_vals == [[0.1],[0.2],[0.2],[0.2, 0.22, 0.24, 0.25, 0.26],[0.1]]
    assert ctrl_vals == [[0.1],[0.2],[0.2],[0.26, 0.2, 0.22, 0.25, 0.24],[0.1]]

# reshape_potential works 
def test_reshape_potential(ctrl_data_input_path, ctrl_data):
    
    ctrl_data = pot.process_nextnano.import_folder(ctrl_data_input_path
                                                    ,file_import_display=True)

    z = 0.45
    f_name = 'Uxy'
    
    # slice an x-y plane of the potentials
    potential2D = pot.reshape_potential(ctrl_data[0]['ctrl_values'],ctrl_data[0]['coord']['x']
        ,ctrl_data[0]['coord']['y'],ctrl_data[0]['coord']['z'],z, f_name)

    assert np.shape(potential2D) == (103,35)

# Make sure xy_potential and write_data works
def test_finilized_2D_data(ctrl_data_input_path, ctrl_data_output_path, ctrl_data):

    z = 0.45
    f_name = ['Uxy']
    # f_name = 'pot'

    file_trig, coords_and_pot = pot.write_data(ctrl_data_input_path, 
        ctrl_data_output_path, z,f_name)


    # Make sure x/y coordinates are added to 2D potential
    assert np.shape(coords_and_pot) == (36,104)

    # Make sure data was converted correctly or saved
    assert file_trig == 0