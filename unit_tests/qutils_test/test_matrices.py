"""
Unittests for the module qutils.matrices.py.

@author: Zach
"""

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

import numpy as np
import pytest

import qudipy.qutils.matrices as matr


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

original_dir = os.path.dirname(os.getcwd())
print(original_dir)

# This will be used to access a tutorial unitary object file
op_data_dir = os.path.join(original_dir,'unit_tests','qutils_test')

# Change working directory to the tutorial data directory for loading/saving
# files
os.chdir(op_data_dir)

#### GLOBAL VARIABLES ####

expect_op_lib = {'PAULI_X': np.array([[0, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0]], dtype=complex),
                'PAULI_Y': np.array([[0, -1.0j, 0, 0],
                                    [1.0j, 0, 0, 0],
                                    [0, 0, 0, -1.0j],
                                    [0, 0, 1.0j, 0]], dtype=complex),
                'PAULI_Z': np.array([[1.0, 0, 0, 0],
                                    [0, -1.0, 0, 0],
                                    [0, 0, 1.0, 0],
                                    [0, 0, 0, -1.0]], dtype=complex),
                'SIGMA_PLUS': np.array([[0, 2, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 2],
                                    [0, 0, 0, 0]], dtype=complex),
                'SIGMA_MINUS': np.array([[0, 0, 0, 0],
                                    [2, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 2, 0]], dtype=complex),
                'UNIT': np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1]], dtype=complex),
                'E_UP': np.array([[1, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 0]], dtype=complex),
                'E_DOWN': np.array([[0, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1]], dtype=complex),
                'CNOT': np.array([[1, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 1, 0]], dtype=complex),
                'SWAP': np.array([[1, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 1, 0]], dtype=complex),
                'SIGMA_PRODUCT': np.array([[3, 0, 0, 0],
                                    [0, 3, 0, 0],
                                    [0, 0, 3, 0],
                                    [0, 0, 0, 3]], dtype=complex),
                'RSWAP': np.array([[2.25-0.25j, 0, 0, 0],
                                    [0, 2.25-0.25j, 0, 0],
                                    [0, 0, 2.25-0.25j, 0],
                                    [0, 0, 0, 2.25-0.25j]], dtype=complex)
                                    }

#### HELPER FUNCTIONS ####

# Function that compares if two objects are equal
def cmp_dict(obj1, obj2):
    try:
        np.testing.assert_array_equal(obj1, obj2)
        return True
    except:
        return False

def get_op(obj):
    N = 2
    k = 2
    ctrl = 2
    trgt = 2
    k1 = 2
    k2 = 2

    # Get hard coded operator method names
    method_list = [method for method in dir(matr.Operator) if method.isupper()]

    # Use list of operator names to change input variables
    input_0 = ['UNIT']
    input_1 = ['PAULI_X', 'PAULI_Y','PAULI_Z','SIGMA_PLUS','SIGMA_MINUS', 
                'E_UP','E_DOWN']
    input_2p1 = ['CNOT']
    input_2p2 = ['SWAP','SIGMA_PRODUCT','RSWAP']

    # Loop over coded operator array outputs and compare to the expected 
    # operater arrays
    for op_key in method_list:
        if op_key in input_0:
            op = getattr(obj, op_key)(N)
            assert cmp_dict(op, expect_op_lib[op_key])
        elif op_key in input_1:
            op = getattr(obj, op_key)(N, k)
            assert cmp_dict(op, expect_op_lib[op_key])
        elif op_key in input_2p1:
            op = getattr(obj, op_key)(N, ctrl, trgt)
            assert cmp_dict(op, expect_op_lib[op_key])
        elif op_key in input_2p2:
            op = getattr(obj, op_key)(N, k1, k2)
            assert cmp_dict(op, expect_op_lib[op_key])

#### TEST FUNCTIONS ####

# Test data is being saved correctly
@pytest.mark.op_lib_manipulation
def test_save_op_dict(data_dir):
    print(data_dir)
    assert os.path.exists(data_dir) == 1
    
# Test saved data is being loaded correctly
def test_load_oplib(data_dir, op_save):

    # Define operator library object
    op = matr.Operator(filename=data_dir)

    # Compare the loaded dictionary to the saved dictionary
    for key in op_save:
        assert cmp_dict(op_save[key], op.lib[key])

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

# Test that operator library object's attribute is being tracked correctly
def test_dict_attr(data_dir, non_unitary_op):

    # Define operator library object
    op = matr.Operator(filename=data_dir)

    # Original attribute state should be true
    assert op.is_unitary

    op.add_operators(non_unitary_op)

    # Attribute should be false after added a non-unitary operator
    assert not op.is_unitary

    op.remove_operators(non_unitary_op)

    # Removing all non-unitary operators should leave attribute as true
    assert op.is_unitary
    
def test_coded_op(op_f, op_d, op_df):

    # Compare the hard coded operators arrays with the expected operator arrays

    # 1) Existing operator dictionary was specified when the object was
    # instantiated, but not operator library object file was given
    get_op(op_d)

    # 2) No existing operator dictionary was specified when the object was
    # instantiated, but an operator library object file was given
    get_op(op_f)

    # 3) Existing operator dictionary was specified when the object was
    # instantiated, and an operator library object file was given
    get_op(op_df)