"""
Unittests for the module potential.process_nextnano.py.

@author: Zach
"""

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import unittest
import numpy as np

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
'''

class ClassTest(unittest.TestCase):

    # Test data is being imported correctly
    def test_data_importation(self):

        # Check that any data from path is imported
        input_nextnano = os.path.dirname(os.getcwd()) + '//tutorials//QuDiPy tutorial data//Nextnano simulations'
        potential = pot.process_nextnano.import_folder(input_nextnano, option=True)
        self.assertTrue(potential)

        # Make sure every trial run was imported
        print(len(potential))
        self.assertEqual(len(potential),5)

        # Make sure a trial run was imported correctly

        # first sublist: control values
        self.assertEqual(len(potential[0][0]),5)

        # second sublsit: potential list containing 3D information
        self.assertEqual(len(potential[0][1]),281190)

        # third sublist: cordinates for potential list
        self.assertEqual(len(potential[0][2]),3)

        # Make sure the structure of the data is correct
        self.assertIsInstance(potential[0][0], list)
        self.assertIsInstance(potential[0][1], np.ndarray)
        self.assertIsInstance(potential[0][2], tuple)

    # parse_ctrl_items works and retrieve_ctrl_vals works
    def test_ctrl_val_minipulation(self):
        input_nextnano = (os.path.dirname(os.getcwd()) 
            + '//tutorials//QuDiPy tutorial data//Nextnano simulations')
        potential = pot.process_nextnano.import_folder(input_nextnano
            ,option=True)

        # Make sure control names are parsed correctly
        for subdir, _, _ in os.walk(input_nextnano):

            # parse voltage information for the directory one level higher than
            # /output
            if subdir != input_nextnano and subdir[-6:] == 'output':
                gates = pot.parse_ctrl_items(subdir,'name')
                break

        self.assertEqual(gates,['V1','V2','V3','V4','V5'])
        
        # Make sure control values are combined correctly
        ctrl_vals = pot.process_nextnano.retrieve_ctrl_vals(potential)
        self.assertEqual(ctrl_vals
            ,[[0.1],[0.2],[0.2],[0.2, 0.22, 0.24, 0.25, 0.26],[0.1]])

    # reshape_potential works 
    def test_reshape_potential(self):
        
        input_nextnano = (os.path.dirname(os.getcwd()) 
            + '//tutorials//QuDiPy tutorial data//Nextnano simulations')
        potential = pot.process_nextnano.import_folder(input_nextnano
                                                        ,option=True)

        coord = potential[0][2]
        z = coord[2][62]
        f_name = 'Uxy'
        
        # slice an x-y plane of the potentials
        potential2D = pot.reshape_potential(potential[0][1],potential[0][2][0]
            ,potential[0][2][1],potential[0][2][2],z, f_name)

        self.assertEqual(np.shape(potential2D), (103,35))

    # Make sure xy_potential and write_data works
    def test_finilized_2D_data(self):
        
        input_nextnano = (os.path.dirname(os.getcwd()) 
            + '//tutorials//QuDiPy tutorial data//Nextnano simulations')
        output_preprocessed = (os.path.dirname(os.getcwd()) 
            + '//tutorials//QuDiPy tutorial data//Pre-processed ' 
            + 'potentials//Pre-processed_data')

        potential = pot.process_nextnano.import_folder(input_nextnano
                                                        ,option=True)

        coord = potential[0][2]
        z = coord[2][62]
        f_name = ['Uxy']

        file_trig, coords_and_pot = pot.write_data(input_nextnano,output_preprocessed
            ,z,f_name)

        # Make sure x/y coordinates are added to 2D potential
        self.assertEqual(np.shape(coords_and_pot),(36,104))

        # Make sure data was converted correctly or saved
        self.assertEqual(file_trig,0)