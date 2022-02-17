"""
Functions for converting 3D nextnano data to 2D data along a user-defined
z-coordinate.

@author : Kewei, Zach
"""

import os, sys
sys.path.append(os.path.abspath(os.path.join(__file__ ,"../../..")))

import qudipy.utils.helpers as hp
import numpy as np
import pandas as pd
import re

def load_file(filename):
    '''
    This function loads data files from a user defined nextnano simulation
    directory.

    Parameters
    ----------
    filename  : String
        full path for the files of interest

    Returns
    -------
    x, y, z  : Tuple of Lists
        a single array ordered by the coordinates for potential.dat files or
        a tuple of 3 element, x, y, z for potential.coord files
    '''

    def mesh(index):
        '''
        This function parses integers in a string which only contains one 
        integer surrounded by white space and/or new line charaters

        Parameters
        ----------
        index  : Integer
            Index value of a string to evaluate from a list of strings
        '''
        return [int(i) for i in d[index].split() if i.isdigit()][0]

    # Import .dat data
    if filename[-4:] == '.dat':
        df = pd.read_csv(filename, header=None)
        data = df.to_numpy()

        return data

    # Import .coord data
    if filename[-6:] == '.coord':

         # Read in xyz dimensions from .fld file for extracting .coord data
        with open(filename.replace('.coord','.fld'), 'r') as f:
            d = f.readlines()

            # Get the dim value for each coordinate axis which corresponds to
            # number of mesh points along each axis            
            xmesh = mesh(3)
            ymesh = mesh(4)
            zmesh = mesh(5)

        # Extract xyz coordinate data
        with open(filename, 'r') as f:
            d = f.readlines()

            # Convert list of strings to list of floats
            data = []
            for i in list(filter(lambda x : x != '\n', d)):
                data.append(float(i.strip()))

            x = data[:xmesh]
            y = data[xmesh:xmesh+ymesh]
            z = data[xmesh+ymesh:xmesh+ymesh+zmesh]

            return x, y, z

def parse_ctrl(filename, ctrl_type):
    '''
    This function collects all of the control items in the filename string and
    processed nextnano potential files are assumed to follow the syntax:
    'TYPE_C1NAME_C1VAL_C2NAME_C2VAL_..._CNNAME_CNVAL.txt'
    where TYPE = 'Uxy' or 'Ez'.
    Refer to tutorial for a more explicit example.

    Parameters
    ----------
    filename : String
        Full path to the filename and containts control items
        (either name/value) information.
    ctrl_type : String
        Specifies what control item needs to be returned. Acceptable
        arguments include ['value','values','name','names'] where case is
        not relevant.

    Returns
    -------
    ctrls : List
        List of control items of the specific control type
    '''

    # Parse string via _,\, /
    parsed_filename = re.split(r'[_/\\]',filename)

    ctrls = []
    for idx, val in enumerate(parsed_filename):
        try:
            if val[0] == 'V':
                if ctrl_type.lower() in ['name', 'names']:
                    ctrls.append(parsed_filename[idx])
                elif ctrl_type.lower() in ['value', 'values']:
                    ctrls.append(float(parsed_filename[idx+1]))
        except:
            pass
    return ctrls

def import_dir(folder, show_files=False):
    '''
    Parameters
    ----------
    folder : String
        Name of the folder where nextnano files are stored


    Keyword Arguments
    -----------------
    show_files : Bool
        Display command line output of the files being imported.

    Returns
    -------
    data : Dictionary
        Where each key/value pair is a key label for the type of data stored in
        value i.e  ctrl_names for a list of voltage labels, ctrl_values for a 
        list of potentials, and coord for a dictionary containing coordinate 
        data.

    nextnano file structure:

    /simulation_runs
        /simulation_run_#_with_gate_voltages
            /output directory
                /data files

    '''

    # Dictionary which holds all of the simulation run data grouped by run
    data = {}

    # Loop over all subdirectories, base directories (ignored), files in
    # folder 
    count = 0   
    for subdir, _, files in os.walk(folder):
        
        # Dictionary containing run data i.e. voltages, potential, 3-tuple of
        # coordinates
        data_per_run = {    
                            'ctrl_names' : {},
                            'ctrl_values' : {},
                            'coord': {  
                                        'x' : {},
                                        'y' : {},
                                        'z' : {}
                                    }
                        }

        # Parse voltage information for the directory one level higher than
        # /output

        # TODO : Standardize nextnano output directory/file structure and naming convention
        if subdir != folder and subdir[-6:] == 'output':
                
            # Elect to display the files being imported from nextnano
            if show_files == True:
                print('Importing .coord and .dat data files from {}:'.format(
                    subdir.replace(str(folder),'')))

            voltage = parse_ctrl(subdir,'value')

            # First append control values
            data_per_run['ctrl_names'] = voltage

            # Second append potential data
            for file in files:

                filename = os.path.join(subdir, file)

                if filename[-13:] == 'potential.dat':
                    data_per_run['ctrl_values'] = load_file(filename)

            # Finally append coordinate data
            for file in files:
                filename = os.path.join(subdir, file)

                if filename[-15:] == 'potential.coord':
                    coord = load_file(filename)
                    data_per_run['coord']['x'] = coord[0]
                    data_per_run['coord']['y'] = coord[1]
                    data_per_run['coord']['z'] = coord[2]
        
            data[count] = data_per_run
            count += 1
    return data

def get_ctrl_vals(nextnano_data):
    '''
    This function takes the dictionary containing all data related to the nextnano
    simulation runs which are to be used to create an interpolation object and
    retuns a list containing list of every unique control value for the given
    control name.

    Parameters
    ----------
    nextnano_data : List
        A list containing lists of voltages, potentials, and coordinates for
        all simulation runs.

    Returns
    -------
    voltages : List
        List of all unique control values per control name for all simulation
        runs in the data directory.

    Example :

        simulation runs:
        [0.1, 0.2, 0.2, 0.20, 0.1]
        [0.1, 0.2, 0.2, 0.22, 0.1]
        [0.1, 0.2, 0.2, 0.24, 0.1]
        [0.1, 0.2, 0.2, 0.25, 0.1]
        [0.1, 0.2, 0.2, 0.26, 0.1]

        unique voltages per gate:
        V1 = [0.1]
        V2 = [0.2]
        V3 = [0.2]
        V4 = [0.2, 0.22, 0.24, 0.25, 0.26]
        V5 = [0.1]

        output:
        [V1, V2, V3, V4, V5]

    '''

    # Store unique voltages per gate
    voltages = []

    # Loop over all gate voltages
    for i in range(len(nextnano_data[0]['ctrl_names'])):

        # Initialize with voltage of the ith gate's first simulation run
        gate = [nextnano_data[0]['ctrl_names'][i]]

        # Loop over all simulation runs
        for j in range(len(nextnano_data)-1):

            # Store voltage if unique
            if (nextnano_data[j]['ctrl_names'][i] 
                != nextnano_data[j+1]['ctrl_names'][i]):
                gate.append(nextnano_data[j+1]['ctrl_names'][i])

        # Convert between list/dict to remove any duplicate voltages per gate
        # while preserving original list order
        gate = list(dict.fromkeys(gate).keys())

        voltages.append(gate)
    return voltages

def reshape_field(potential, x, y, z, f_type, slice=None, show_z=None):
    '''
    This function reshapes the 1d array of potential data according to the
    number of data points for the x,y, and z coordinates then retuns the field
    along the XY-plane at the z-coordinate specified by slice.

    Parameters
    ----------
    potential : Array
        An array conaining all of the potential data for the x,y, and z coordinates.

    x,y,z : List
        Coordinates for the potential data contained in the 1d potential array.

    f_type : List
        Field type identifier either ['field', 'electric', 'Ez']  or
        ['pot', 'potential', 'Uxy'] where case is not relevant.

    Keyword Arguments
    -----------------
    slice : Float
        z coordinate value to generate a 2D potential interpolation object
        from all simulation runs or 3D array if slice is unspecified.

    show_z : Bool
        A flag to display what the actually z coordinate which the 3D data was
        sliced along.

    Returns
    -------
    field2DArray : Array
        2d array of the potentials in the XY-plane along slice z-coordinate.

    '''

    # Find the index for the closest coordinate which corresponds to the desired
    # z-coordinate
    index_tuple,nearest_z = hp.find_nearest(z,slice)
    index = index_tuple[0]

    if show_z:
        print(f'The nearest z coordinate to slice={slice}, is z={nearest_z}.')

    # Number of data points per axis provided from simulations
    xsize = len(x)
    ysize = len(y)
    zsize = len(z)

    pot3DArray = np.reshape(potential,(xsize,ysize,zsize), order='F')

    if f_type.lower() in ['field', 'electric', 'ez']:
        gradient = np.gradient(pot3DArray,x,y,z)[-1]
        field2DArray = gradient[:, :, index]
    elif f_type.lower() in ['pot', 'potential', 'uxy']:
        field2DArray = pot3DArray[:, :, index]

    return field2DArray

def xy_pot(potential, gates, slice, f_type, output_dir_path=None, save=None):
    '''
    This function takes the potential data and control value information and
    saves a 2D potential cross section along slice to a text file in the user
    specified directory for a given field type.

    Parameters
    ----------
    potential : Dictionary
        Where each key/value pair is a key label for the type of data stored in
        value i.e  ctrl_names for a list of voltage labels, ctrl_values for a 
        list of potentials, and coord for a dictionary containing coordinate 
        data.

    gates : List
        List of lists for the unique control values for each control name.

    slice : Float
        Z-coordinate to take a cross section on.

    Keyword Arguments
    ----------
    f_type : List
        Field type identifier
        ['field', 'electric', 'Ez', 'pot', 'potential', 'Uxy'] where case is not
         relevant.

    output_dir_path : String
        Directory path for the processed nextnano field data.

    Returns
    -------
    file_trig : Integer
        Returns 0 if Potential or electric field XY-plane data is saved as a
        text file for the z-coordinate along slice. Otherwise, -1 is returned.

    coord_and_pot : Array
        Returns the 2D pontenial data slice with the x/y coordinates appended to
        the first row and column.
    '''

    potential_copy = potential.copy()
    # Loop through each combination of gate voltages
    for i in potential_copy:

        if f_type.lower() in ['pot', 'potential', 'uxy']:
            # Capital U used here for writing nextnano data to text file
            f_name = 'Uxy'
            # Slice an x-y plane of the potentials
            potential2D = reshape_field(potential_copy[i]['ctrl_values'], 
                                            potential_copy[i]['coord']['x'],
                                            potential_copy[i]['coord']['y'], 
                                            potential_copy[i]['coord']['z'],
                                            f_name, slice)
        elif f_type.lower() in ['field', 'electric', 'ez']:
            # Capital E used here for writing nextnano data to text file
            f_name = 'Ez'
            potential2D = reshape_field(potential_copy[i]['ctrl_values'], 
                                            potential_copy[i]['coord']['x'],
                                            potential_copy[i]['coord']['y'], 
                                            potential_copy[i]['coord']['z'],
                                            f_name, slice)

        # Create an array of zeros with the dimension of the potential 2D slice
        # and x/y coordinate axis
        coords_and_pot = np.zeros((np.shape(potential2D)[0]+1,
            np.shape(potential2D)[1]+1),dtype=float)

        # Insert x,y, and potential 2D slice into array
        coords_and_pot[1:,0] = potential_copy[i]['coord']['x']
        coords_and_pot[0,1:] = potential_copy[i]['coord']['y']
        coords_and_pot[1:,1:] = -1*potential2D

        # Transpose array to make the y-axis data run row wise and x-axis data
        # run column wise.
        coords_and_pot = np.transpose(coords_and_pot)

        j = 0    
        while j < len(potential_copy[i]['ctrl_names']):
            f_name = (f_name + '_' + gates[j] + '_' +
                "{:.3f}".format(potential_copy[i]['ctrl_names'][j]))
            j += 1

        f_name +='.txt'

        # Create directory for preprocessed data
        if not os.path.exists(output_dir_path):
            os.mkdir(output_dir_path)

        # Join file name to directory path
        f_path = os.path.join(output_dir_path,f_name)

        if save:
        # Try to save the data to a text file
            try:
                # Save potential data for xy slice
                np.savetxt(f_path, coords_and_pot, delimiter=',')
                file_trig = 0
            except:
                file_trig = -1
        else:
            # Return none if no save value was provided
            file_trig = None

    return file_trig, coords_and_pot

def write_data(input_dir_path, output_dir_path, slice, f_type):
    '''
    This function takes the data directory containing the 3D nextnano simulation
    run files and calls helper functions which process the 3D data into 2D field
    data along a specific slice cross section. Then 2D field information is
    saved in an output directory for every user defined field type.

    Parameters
    ----------
    input_dir_path : String
        Directory path for the pre-processed nextnano field data files.

    output_dir_path : String
        Directory path to write the processed nextnano field data files.

    slice : Float
        Z coordinate value to generate a 2D potential interpolation object
        for from all simulation runs.

    Keyword Arguments
    ----------
    f_type : List
        Field type identifier either ['field', 'electric', 'Ez']  or
        ['pot', 'potential', 'Uxy' ] where case is not relevant.

    Returns
    -------
    file_trig : Integer
        Returns 0 if Potential or electric field XY-plane data is saved as a
        text file for the z-coordinate along slice. Otherwise, -1 is returned.
    '''
    potential = import_dir(input_dir_path)

    # Find nearest slice from user specified slice
    _, nearest_slice = hp.find_nearest(potential[0]['coord']['z'], slice)

    for subdir, _, _ in os.walk(input_dir_path):

        # Parse voltage information for the directory one level higher
        # than /output
        if subdir != input_dir_path and subdir[-6:] == 'output':
            gates = parse_ctrl(subdir,'name')
            break

    output_dir_path = output_dir_path + '_for_nearest_slice{:.3e}'.format(nearest_slice)

    # Write xy potential files
    for i in f_type:

        # Try to write xy potential text files
        file_trig, _ = xy_pot(potential,gates,nearest_slice,i
            ,output_dir_path, save=True)

        # Indicate to user that the file failed/succeeded writting the data
        # to a text file
        if file_trig == 0:
            print('SAVE SUCCESS: Converting 3D nextnano simulation data to 2D ' 
            'XY-plane {} along slice for z = {}.'.format(i,nearest_slice))
        elif file_trig == -1:
            print('FAILED SAVE: Failed to convert 3D nextnano simulation data '
                'too 2D XY-plane {} data along slice for '
                + 'z = {}.'.format(i,nearest_slice))
        else:
            print('NO DATA SAVED: Save flag in xy_pot is set to None or False')

    return file_trig
