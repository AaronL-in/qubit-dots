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
    assert len(ctrl_data) == 8

    # Make sure a trial run was imported correctly

    # first sublist: control names
    assert len(ctrl_data[0]['ctrl_names']) == 6

    # second sublsit: potential list containing 3D information
    assert len(ctrl_data[0]['ctrl_values']) == 376656

    # third sublist: cordinates for potential list
    assert len(ctrl_data[0]['coord']) == 3

    # Make sure the structure of the data is correct
    assert type(ctrl_data[0]['ctrl_names']) is list
    assert type(ctrl_data[0]['ctrl_values']) is np.ndarray
    assert type(ctrl_data[0]['coord']) is dict

# get_ctrl works and get_ctrl_vals works
def test_ctrl_val_minipulation(ctrl_data_input_path, ctrl_data):
    
    # Make sure control names are parsed correctly
    for subdir, base_dir, files in os.walk(ctrl_data_input_path):

        # Collect simulation meta data files
        list_of_files = pot.get_files(ctrl_data_input_path, '.log')

        # parse voltage information for the directory one level higher than
        # /output or /bias_000_000_000
        trig = False
        for file in files:
            if subdir != ctrl_data_input_path and file in list_of_files:
                gates = pot.get_ctrl(os.path.join(subdir,file),'name')
                # Trigger inner loop break
                trig = True
                break
        
        # Break outer loop if inner loop was broken
        if trig == True:
            break

    assert gates == ['V3','V2','V4','V5','V1','Si']
    
    # Make sure control values are combined correctly
    ctrl_vals = pot.process_nextnano.get_ctrl_vals(ctrl_data)

    print(ctrl_data.keys)
    print(ctrl_vals)

    assert ctrl_vals == [[0.2],[0.2, 0.4],[0.2, 0.22, 0.24, 0.26], [0.1], [0.1], [0.0]]


#### THESE UNIT TESTS ARE BEING DEPRECATED WITH THE REFACTORED QUDIPY CODE ####

# # reshape_field works 
# def test_reshape_field(ctrl_data_input_path, ctrl_data):
    
#     ctrl_data = pot.process_nextnano.import_dir(ctrl_data_input_path
#                                                     ,show_files=True)

#     z = 0.45
#     f_name = 'Uxy'

#     # slice an x-y plane of the potentials
#     potential2D = pot.reshape_field(ctrl_data[0]['ctrl_values'],ctrl_data[0]['coord']['x']
#         ,ctrl_data[0]['coord']['y'],ctrl_data[0]['coord']['z'],z, f_name)

#     assert np.shape(potential2D) == (103,35)

# # Make sure xy_potential and write_data works
# def test_finilized_2D_data(ctrl_data_input_path, ctrl_data_output_path, ctrl_data):

#     z = 0.45
#     f_name = ['Uxy']
#     # f_name = 'pot'

#     file_trig, coords_and_pot = pot.write_data(ctrl_data_input_path, 
#         ctrl_data_output_path, z,f_name)


#     # Make sure x/y coordinates are added to 2D potential
#     assert np.shape(coords_and_pot) == (36,104)

#     # Make sure data was converted correctly or saved
#     assert file_trig == 0