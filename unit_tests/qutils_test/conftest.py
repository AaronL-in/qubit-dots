import os, sys
from pyparsing import opAssoc
sys.path.append(os.path.dirname(os.getcwd()))
import pytest
import numpy as np

# import any packages used for testing the module
import qudipy.qutils.matrices as matr

'''
@pytest.fixture(scope="module")
def <qutils_test global variable>():
    
    <some code> 
    
    return <some variable>


    NOTE: 'scope="module"' gives the fixtures in the file scope to the 
        <test dir name>_test only. If the scope for a fixture is needed
        for all subdirectories of unit_test, then modify conftest.py
        in the directory one level higher.
'''


# Dictionary to initialize ops object with
@pytest.fixture(scope="module")
def op_save():
    ops = {
        'PAULI_X': np.array([[0, 1], [1, 0]], dtype=complex),
        'PAULI_Y': np.array([[0, -1.0j], 
            [1.0j, 0]],dtype=complex),
        'PAULI_Z': np.array([[1, 0], [0, -1]], dtype=complex),
        'PAULI_I': np.array([[1, 0], [0, 1]], dtype=complex),
        'PAULI_I_4x4': np.array([[1, 0, 0, 0], 
                                [0, 1, 0, 0], 
                                [0, 0, 1, 0], 
                                [0, 0, 0, 1]], dtype=complex)
    }

    return ops

# Fixture which generates an operator library object and save it to 
# unit test>qutils test directory     
@pytest.fixture(scope="module")
def data_dir(op_save):

    # Get curentl working directory
    original_dir = os.path.dirname(os.getcwd())

    # This is used to access operator_test unitary object file
    op_data_dir = os.path.join(original_dir,'qutils_test')

    # Define operator library object
    ops = matr.Operator(operators=op_save)

    # Specify filename for unitary operators object
    temp_data = 'operator_test.npz'

    # Save the operators in op_save to an object file
    dir = os.path.join(op_data_dir, temp_data)
    ops.save_ops(temp_data)

    # yield is used to make sure the save object file is removed after test is
    # complete
    yield dir
    os.remove(dir)
    
# Define non-unitary dictionary    
@pytest.fixture(scope="module")
def non_unitary_op():
        return {
        'added_op': np.array([[-1.0+1.0j,-1.0+1.0j],[-1.0+1.0j,-1.0+1.0j]])
    }

# Define operator object from object file   
@pytest.fixture(scope="module")
def op_f(data_dir):
        return matr.Operator(filename=data_dir)
        
# Define operator object from existing dictionary of operators  
@pytest.fixture(scope="module")
def op_d(op_save):
        return matr.Operator(operators=op_save)

# Define operator object from object file and existing dictionary of operators
@pytest.fixture(scope="module")
def op_df(op_save, data_dir):
        return matr.Operator(operators=op_save, filename=data_dir)


