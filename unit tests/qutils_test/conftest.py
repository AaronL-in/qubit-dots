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


@pytest.fixture(scope="module")
def op_save():
    # Dictionary to initialize ops object with
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

