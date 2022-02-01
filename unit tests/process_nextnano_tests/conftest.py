import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
import pytest

# import any packages used for testing the module
import qudipy.potential as pot

# Add fixtures to be used accross the given package of the module
@pytest.fixture(scope="module")
def ctrl_data_input_path():
    # Check that any data from path is imported
    # input = (os.path.join(sys.path[0],'QuDiPy tutorial data','Nextnano simulations','1_dot_P_0p5_0p7'))
    input = (os.path.join(os.path.dirname(sys.path[0]),'tutorials','QuDiPy tutorial data','Nextnano simulations'))

    return input

@pytest.fixture(scope="module")
def ctrl_data_output_path():
    # Check that any data from path is imported
    output = (os.path.join(os.path.dirname(sys.path[0]),'tutorials','QuDiPy tutorial data','Pre-processed_data'))
        
    return output

@pytest.fixture(scope="module")
def ctrl_data(ctrl_data_input_path):
    # Check that any data from path is imported
    # input_nextnano = (os.path.join(sys.path[0],'QuDiPy tutorial data','Nextnano simulations'))
    data = pot.process_nextnano.import_folder(ctrl_data_input_path, file_import_display=True)

    return data