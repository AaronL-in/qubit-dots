"""
Unittests for the module potential.process_nextnano.py.

@author: Zach
"""

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

print(os.path.dirname(os.getcwd()))

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
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

original_dir = os.path.dirname(os.getcwd())
print(original_dir)

# This will be used to access a tutorial unitary object file
op_data_dir = os.path.join(original_dir,'unit tests','qutils_test')

print(op_data_dir)

import qudipy.qutils.matrices as matr

import numpy as np
import matplotlib.pyplot as plt

# Initialize unitary object:

# change working directory to the tutorial data directory for loading/saving
# files
os.chdir(op_data_dir)

# Test data is being saved correctly
@pytest.mark.op_lib_monipulation
# @pytest.fixture
def test_save_op_dict(data_dir):
    assert os.path.exists(data_dir) == 1
    
# Test saved data is being loaded correctly
def test_load_oplib(data_dir, op_save):

    # Define operator library object
    op = matr.Operator(filename=data_dir)

    # Compare the loaded dictionary to the saved dictionary
    assert(op.lib['PAULI_X'] == op_save['PAULI_X']).all() 
    assert(op.lib['PAULI_Y'] == op_save['PAULI_Y']).all() 
    assert(op.lib['PAULI_Z'] == op_save['PAULI_Z']).all() 
    assert(op.lib['PAULI_I'] == op_save['PAULI_I']).all() 
    assert(op.lib['PAULI_I_4x4'] == op_save['PAULI_I_4x4']).all() 

# Test operators are added/remove from operator object library
def test_add_remove_oplib(data_dir, non_unitary_op):

    # Define operator library object
    op = matr.Operator(filename=data_dir)

    # Add the an operator
    op.add_operators(non_unitary_op)

    # Test if operator was added
    assert(op.lib['added_op'] == non_unitary_op['added_op']).all() 

    # Remove the just added operator
    op.remove_operators(non_unitary_op)

    # Test if operator was removed
    assert 'added_op' not in op.lib.keys() 

# Test that oparator library object's attribute is being tracked correctly
def test_dict_attr(data_dir, non_unitary_op):

    # Define operator library object
    op = matr.Operator(filename=data_dir)

    # Original attribute state should be true
    assert op.is_unitary == True

    op.add_operators(non_unitary_op)

    # Attribute should be false after added a non-unitary operator
    assert op.is_unitary == False

    op.remove_operators(non_unitary_op)

    # Removing all non-unitary operators should leave attribute as true
    assert op.is_unitary == True
    


