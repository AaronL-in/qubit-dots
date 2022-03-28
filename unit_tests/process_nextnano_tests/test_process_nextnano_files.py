'''
Unittests for the module potential.process_nextnano.py.

@author: Zach
'''

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pytest

import qudipy.potential as pot
import qudipy.utils.helpers as hp

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

    assert len(ctrl_data) == 4

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

# get_ctrl works and retrieve_ctrl_vals works
def test_ctrl_val_minipulation(ctrl_data_input_path, ctrl_data):
    
    # Make sure control names are parsed correctly
    
    # Collect simulation meta data file names to parse
    list_of_files = pot.get_files(ctrl_data_input_path)

    # Loop over all sub_directories, base directories, files in
    # dir
    for sub_dir, base_dir, files in os.walk(ctrl_data_input_path):

        # First move into a subdirectory
        if sub_dir != dir and base_dir == ['output']:

            # Search for simulation run meta data file
            for file in files:
                # If the file name is the same as one of the directories then 
                # get the ctrl_item information
                if file in list_of_files:
                    
                    # Generate absolute path to file that is names the same as
                    # the directory
                    filename = os.path.join(sub_dir, file)

                    # Get voltages
                    gates = pot.get_ctrl(filename, ctrl_type='name')

                    # Check that all simulation runs have the same gate names
                    assert gates == ['V1','V2','V3','V4','V5']

    
    # Make sure control values are combined correctly
    ctrl_vals = pot.process_nextnano.get_ctrl_vals(ctrl_data)

    # Check that the expected gate voltages were extracted from the nextnano
    # meta data file
    assert ctrl_vals == [[0.1],[0.2],[0.2],[0.2, 0.22, 0.24, 0.26],[0.1]]

# reshape_field works 
def test_reshape_field(ctrl_data_input_path, ctrl_data):
    
    ctrl_data = pot.process_nextnano.import_dir(ctrl_data_input_path
                                                    ,show_files=True)

    z = 0.45
    f_name = 'Uxy'
    
    print(ctrl_data.keys())
    print(ctrl_data[0].keys())

    # slice an x-y plane of the potentials
    potential2D = pot.reshape_field(ctrl_data[0]['ctrl_values'],
                                    ctrl_data[0]['coord']['x'], 
                                    ctrl_data[0]['coord']['y'], 
                                    ctrl_data[0]['coord']['z'],
                                    f_name, z)

    assert np.shape(potential2D) == (103,35)

# Make sure xy_potential and write_data works
def test_finilized_2D_data(ctrl_data_input_path, ctrl_data_output_path, ctrl_data):

    z = 0.45
    f_name = ['Uxy']

    # Import 3D potential data to be processed
    potential = pot.import_dir(ctrl_data_input_path)

    # Collect simulation meta data files
    list_of_files = pot.get_files(ctrl_data_input_path)

    # Find nearest slice from user specified slice
    _, nearest_slice = hp.find_nearest(potential[0]['coord']['z'], z)

    for sub_dir, _, files in os.walk(ctrl_data_input_path):

        # Parse voltage information from simulation run meta data file
        for file in files:
            if (sub_dir != ctrl_data_input_path 
                and sub_dir[-6:] != 'output' 
                and file in list_of_files):

                filename = os.path.join(sub_dir,file)
                gates = pot.get_ctrl(filename, ctrl_type='name')
                break

    # Generate the 2D slice containing the coordinate and field data for the 
    # f_name[0] = 'Uxy'
    _, coord_and_pot = pot.xy_pot(potential, gates, nearest_slice, 
                                f_name[0], save=False)

    # Make sure x/y coordinates are added to 2D potential
    assert np.shape(coord_and_pot) == (36,104)

    # Try to save a 2D slice of data
    file_trig = pot.write_data(ctrl_data_input_path, 
                                                ctrl_data_output_path, 
                                                z, f_name)

    # Make sure data was saved
    assert file_trig == True