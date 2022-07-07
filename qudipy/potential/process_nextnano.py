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
from scipy import constants

def index_containing_substring(list_of_strings, substring):

    '''
    This function finds the index for the first string in a list of strings
    which contains the substring.
    
    Parameters
    ----------
    list_of_strings : list
        list of strings

    Returns
    -------
    Index position of string containing substring, otherwise, -1
    '''

    # Check that the data type for list_of_strings is list
    if type(list_of_strings) != list:
        ValueError(f'Entry in substring postion is not of type list, but {type(list_of_strings)}')

    for idx, string in enumerate(list_of_strings):

        # Check that the data types are are strings
        if type(string) != str:
            ValueError(f'Index entry {idx} in list is not of type str, but {type(string)}')
        if type(substring) != str:
            ValueError(f'Entry in substring postion is not of type str, but {type(substring)}')

        # Retrun index of first string containing substring
        if substring in string:
            return idx

    # Return fail code -1 if no substring is found in any of the strings        
    return -1


def get_files(dir, file_type):

    '''
    This function collects the sub-directory names for the main input nextnano
    data directory and appends the file type of interest to the name for 
    later use finding the simulation log file containing meta data about the 
    simulation run.
    
    Parameters
    ----------
    dir : string
        Full path name to directory containing sub-directories of simulation run
        data
    file_type: string
        File type extension to append to sub-directory names i.e. '.log'

    Returns
    -------
    A list of file names base on the sub-directory names
    '''

    # Collect the names of every subdirectory in the dir path
    list_subfolders_with_paths = [f.name for f in os.scandir(dir) if f.is_dir()]

    # Append text document suffix on each string in the list
    list_of_files = [string + file_type for string in list_subfolders_with_paths]

    return list_of_files

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

def get_ctrl(filename, ctrl_type):
    '''
    This function collects all of the control items using the log text file
    generated by nextnano upon the start of a simulation run. The variable names
    and values are written the the line after which
    "STARTING CALCULATION FOR BIAS POINT" is written.

    Parameters
    ----------
    filename : String
        Full path to the filename where the simulation run file is stored.
    ctrl_type : String
        Specifies what control item needs to be returned. Acceptable
        arguments include ['value','values','name','names'] where case is
        not relevant.

    Note that the structure of the variable information in the log file is
    similiar to what follows:

    00:00:03 ========= STARTING CALCULATION FOR BIAS POINT =====================
    00:00:03 V1 0.1 V   V2 0.2 V   V3 0.2 V   V4 0.2 V    V5 0.1 V

    Returns
    -------
    ctrls : List
        List of control items of the specific control type
    '''

    # List of control items
    ctrl_item = []
    # Phrase to search for
    phrase = 'STARTING CALCULATION FOR BIAS POINT'
    # Trigger used to indication  phrase was found
    trig = 0

    # Open the simulation log file
    with open(filename, 'r') as f:
        d = f.readlines()

        # Read individual lines in the file
        for read_line in d:

            # If phrase is found trigger trig by setting the value to 1
            if phrase in read_line:
                trig = 1
            # If the phrase was found the next line in the file will be
            # evaluated
            elif trig == 1:
                # Split the given line in the file up into individual strings
                split_line = read_line.split()
                # Reset trig so that no more lines are evaluated
                trig = 0

                # Loop through strings in list
                for i in range(len(split_line)):

                    # Only append if the index is not a multple of 3 and not
                    # first/last element
                    if i % 3 == 0 and i != 0:
                        # Get control values
                        if ctrl_type in ['value', 'values']:
                                # i-1 is the postion of the control value
                                # relative to i modulo 3 = 0
                                ctrl_item.append(float(split_line[i-1]))
                        # Get control names
                        elif ctrl_type in ['name', 'names']:
                                # i-2 is the postion of the control value
                                # relative to i modulo 3 = 0
                                ctrl_item.append(split_line[i-2])

    return ctrl_item

def import_dir(dir, show_files=False):
    '''
    Parameters
    ----------
    dir : String
        Name of the dir where nextnano files are stored


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

    # Collect simulation meta data files
    list_of_files = get_files(dir, '.log')

    # Dictionary which holds all of the simulation run data grouped by run
    data = {}
    count = 0

    # Loop over all sub_directories, base directories (ignored), files in
    # dir
    for sub_dir, base_dir, files in os.walk(dir):

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

        test1 = sub_dir
        test2 = base_dir
        test3 = files
        combined = '\t'.join(base_dir)

        test4 = 'bias_' in combined

        # Check if /output directory exist
        condional_flag = False

        output_idx = index_containing_substring(base_dir,'output')

        if  output_idx != -1 and base_dir[output_idx] == ['output']:
            output_flag = True
        else:
            output_flag = False

        # First move into a subdirectory
        if  sub_dir != dir and 'bias_' in combined:

            # Check if files one level down in the subdirectory exists
            if not files:
                print(f'WARNING: no individual files in directory {sub_dir}')

            # Search for simulation run meta data file
            for file in files:
                # If the file name is the same as one of the directories then get
                # the ctrl_item information
                if file in list_of_files:

                    # Generate absolute path to file that is names the same as the
                    # directory
                    filename = os.path.join(sub_dir, file)

                    # Get voltages
                    voltages = get_ctrl(filename, ctrl_type='value')


            #TODO: handle when above loop can't find any files matching any file in list_of_files
            # Check if control data in the subdirectory exists
            if not voltages:
                print(f'WARNING: no simulation run file with control data found in directory {sub_dir}')

            # Assign control values for the given simulation run
            data_per_run['ctrl_names'] = voltages

            # Find the index for the /bias_000_000 string in the list of sub-directories
            data_dir_idx = index_containing_substring(base_dir,'_000')

            # Loop through the all files under the /bias_000_000 directory
            for data_sub, _, data_files in os.walk(os.path.join(sub_dir,base_dir[data_dir_idx])):

                test6 = data_sub[-4:] 

                # if data_sub[:5] == '_000':

                # Assign potential and coordinate data
                for file in data_files:

                    # Generate absolute path to file
                    filename = os.path.join(data_sub, file)
                    
                    # if file == 'potential.dat':
                    if filename[-13:] == 'potential.dat':

                        # Elect to display the files being imported from nextnano
                        if show_files == True:
                            print('Importing .coord data files from {}:'.format(
                                sub_dir.replace(str(dir),'')))

                        data_per_run['ctrl_values'] = load_file(filename)

                    # if file == 'potential.coord':
                    if filename[-15:] == 'potential.coord':

                        # Elect to display the files being imported from nextnano
                        if show_files == True:
                            print('Importing .dat data files from {}:'.format(
                                sub_dir.replace(str(dir),'')))

                        coord = load_file(filename)
                        data_per_run['coord']['x'] = coord[0]
                        data_per_run['coord']['y'] = coord[1]
                        data_per_run['coord']['z'] = coord[2]

            # Assign every data set per simulation run to a single list of
            # dictionaries
            data[count] = data_per_run
            count += 1

        # In case the user using nextnano created output data per simulation run
        # which is encapsulated within the /output directory 
        elif sub_dir != dir and output_flag == True:

            # Check if files one level down in the subdirectory exists
            if not files:
                print(f'WARNING: no individual files in directory {sub_dir}')

            # Search for simulation run meta data file
            for file in files:
                # If the file name is the same as one of the directories then get
                # the ctrl_item information
                if file in list_of_files:

                    # Generate absolute path to file that is names the same as the
                    # directory
                    filename = os.path.join(sub_dir, file)

                    # Get voltages
                    voltages = get_ctrl(filename, ctrl_type='value')


            #TODO: handle when above loop can't find any files matching any file in list_of_files
            # Check if control data in the subdirectory exists
            if not voltages:
                print(f'WARNING: no simulation run file with control data found in directory {sub_dir}')

            # Assign control values for the given simulation run
            data_per_run['ctrl_names'] = voltages

            # Loop through the all files under the /output directory
            for data_sub, _, data_files in os.walk(os.path.join(sub_dir,base_dir[output_idx])):

                # Assign potential and coordinate data
                for file in data_files:

                    # Generate absolute path to file
                    filename = os.path.join(data_sub, file)
                    
                    # if file == 'potential.dat':
                    if filename[-13:] == 'potential.dat':

                        # Elect to display the files being imported from nextnano
                        if show_files == True:
                            print('Importing .coord data files from {}:'.format(
                                sub_dir.replace(str(dir),'')))

                        data_per_run['ctrl_values'] = load_file(filename)

                    # if file == 'potential.coord':
                    if filename[-15:] == 'potential.coord':

                        # Elect to display the files being imported from nextnano
                        if show_files == True:
                            print('Importing .dat data files from {}:'.format(
                                sub_dir.replace(str(dir),'')))

                        coord = load_file(filename)
                        data_per_run['coord']['x'] = coord[0]
                        data_per_run['coord']['y'] = coord[1]
                        data_per_run['coord']['z'] = coord[2]

            # Assign every data set per simulation run to a single list of
            # dictionaries
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

    # Collect control values for all runs
    gate = []
    for i in range(len(nextnano_data.keys())):
        gate.append(nextnano_data[i]['ctrl_names'])

    # Convert list of lists to array
    gate = np.array(gate)

    ctrl_vals = []
    for i in range(np.shape(gate)[1]):
        # Append all unique control values per control variable. This is done
        # by: converting array of voltages per gate to set, sort the set items,
        # convert the set to a list, and append the list to ctrl_vals.
        ctrl_vals.append(list(sorted(set(gate[:,i]))))

    return ctrl_vals

def reshape_field(potential, x, y, z, f_type, slice=None, show_z=None):
    '''
    This function reshapes the 1d array of potential data according to the
    number of data points for the x,y, and z coordinates then retuns the field
    along the XY-plane at the z-coordinate specified by slice.

    Parameters
    ----------
    potential : Array
        An array conaining all of the electrostatic potential data for the
        x, y, and z coordinates using the units V/m.

    x, y, z : List
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

        # Calculate the gradient of the electrostatic potential for the 3D data
        gradient = np.gradient(pot3DArray,x,y,z)[-1]

        # Compute the electric field from the electrostatic potential gradient
        # for the given z-slice along the given array index.
        field2DArray = - gradient[:, :, index]

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
    file_trig : Bool
        Returns True if potential or electric field XY-plane data is saved as a
        text file for the z-coordinate along slice. Otherwise, False is
        returned.

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
        coords_and_pot[1:,1:] = potential2D

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
                file_trig = True
            except:
                file_trig = False
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
    file_trig : Bool
        Returns True if potential or electric field XY-plane data is saved as a
        text file for the z-coordinate along slice. Otherwise, False is
        returned.
    '''
    potential = import_dir(input_dir_path)

    # Collect simulation meta data files
    list_of_files = get_files(input_dir_path, '.log')

    # Find nearest slice from user specified slice
    _, nearest_slice = hp.find_nearest(potential[0]['coord']['z'], slice)

    for sub_dir, _, files in os.walk(input_dir_path):

        # Parse voltage information from simulation run meta data file
        for file in files:
            if sub_dir != input_dir_path and sub_dir[-6:] != 'output' and file in list_of_files:

                filename = os.path.join(sub_dir,file)
                gates = get_ctrl(filename, ctrl_type='name')
                break

    output_dir_path = output_dir_path + '_for_nearest_slice{:.3e}'.format(nearest_slice)

    # Write xy potential files
    for i in f_type:

        # Try to write xy potential text files
        file_trig, _ = xy_pot(potential,gates,nearest_slice,i
            ,output_dir_path, save=True)

        # Indicate to user that the file failed/succeeded writting the data
        # to a text file
        if file_trig:
            print('SAVE SUCCESS: Converting 3D nextnano simulation data to 2D '
            'XY-plane {} along slice for z = {}.'.format(i,nearest_slice))
        elif not file_trig:
            print('FAILED SAVE: Failed to convert 3D nextnano simulation data '
                'too 2D XY-plane {} data along slice for '
                + 'z = {}.'.format(i,nearest_slice))
        else:
            print('NO DATA SAVED: Save flag in xy_pot is set to None or False')

    return file_trig