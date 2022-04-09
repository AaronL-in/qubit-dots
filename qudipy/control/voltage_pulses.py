'''
    Module that contains methods that design "simple" one- and two-qubit gates.
    Will return control pulse objects eventually, but now it is used only to 
    find pulse duration or amplitude
'''
##*************************************************
import os, sys
sys.path.append(os.path.dirname(os.getcwd()))
## ************************************************



#TODO:
# search function for delta g = g_0 for all qubits
# 

import numpy as np
import pandas as pd
# TODO UNCOMMENT from qudipy.control import shapes
from qudipy.circuit.control_pulse import ControlPulse
from qudipy.circuit.quantum_circuit import QuantumCircuit
from qudipy.circuit.file_parsers import load_pulses, _load_one_pulse, load_circuit
from scipy.constants import e, m_e

def balance_zeeman(delta_g_interp, v_offset, f_rf):
    '''
    Find the value of Zeeman field that ensures no qubit rotations at idling.

    Parameters
    -----------------
    delta_g_interp: function
        (ideally, but only one interpolating function for now)
        Iterable of interpolating functions \delta g_i(\vec{V}) for 
        all qubits, including inactive ones
    v_offset: Float
        Voltage offset when no pulse is applied.
    f_rf: float
        ESR frequency.

    Requires
    ------------
    * delta_g_interp to take Anyof(Float Int) and return Anyof(Float Int)
    * v_offset to take Anyof(Float Int) and return Anyof(Float Int)

    Returns
    -------
    B_0: float
        The value of Zeeman field that ensures 
        no rotation occurs on any of the idling qubits
    '''
    delta_g_0 = delta_g_interp(v_offset)

    
    omega = 2 * np.pi * f_rf / (1 + delta_g_0 / 2)
    B_0 = m_e / e * omega

    return B_0


def rot(rotation_axis, theta, n_qubits, active_qubits, 
                delta_g_interp, v_unit_shape, B_0, num_val=100):
    '''
    @author: Madi & Others (TODO specify others)

    Chooses the optimal duration for a ROT(\theta) pulse of the specified pulse 
    shape, where the argument 'axis' defines rotation. 
    ***For now*** the function assumes for the sake of simplicity that
    the delta g interpolators are the same for each qubit

    Parameters
    -----------------
    rotation_axis: iterable of length 3 OR string ('X', 'Y', 'Z') 
        Axis of qubit rotation.
    theta: float
        Angle of qubit rotation in **degrees**.
    n_qubits: int
        Total number of qubits.
    active_qubits: tuple/list/array of ints
        Positions of qubits (starting from 1, not 0!) on which the pulse
         is supposed to act. The rest will be made inactive.
    delta_g_interp: function
        (ideally, but only one interpolating function for now)
        Iterable of interpolating functions \delta g_i(\vec{V}) for 
        all qubits, including inactive ones
    v_unit_shape: function
        Shape of a voltage pulse defined over a range of time [0,1], i. e. 
        over the normalized time.
    B_0: float
        Zeeman field

    Requires
    ------------
    * delta_g_interp to take Anyof(Float Int tuple array List) and return 
            Anyof(Float Int tuple array List) -> will return a type of the same shape
    * v_unit_shape to take Anyof(Float Int tuple array List) and return 
            Anyof(Float Int tuple array List) -> will return a type of the same shape

    Keyword Arguments
    -----------------
    num_val: int
        Number of data points during the pulse.  

    #TODO make it possible to specify this value 
    # t_start: float  
    #     Beginning of the system evolution observation. The actual pulse 
    #     (not constant offset) may start before or after this time point.


    Returns
    -------
    rot_pulse: Control pulse object
        Pulse with both effective (delta_g_i) and actual variables
        (V_i, B_0, B_rf, phi)  values to be used in

    #TODO Data frame of pulses V_i(t), B_0, B_rf, phi, ... within 
    # the ControlPulse.ctrl_pulses object

    '''
    # make sure that the rotation_axis has 3 components or is an appropriate axis
    if rotation_axis in ['X', 'x', 'Y', 'y', 'Z', 'z']:
        axis = rotation_axis.upper()
    elif len(rotation_axis) != 3:
        print("Please enter an iterable with 3 Float components or enter one of 'X', 'Y', 'Z'")
    else:
        axis = rotation_axis
    
    # build a tuple of positions of inactive qubits
    all_qubits = range(1, n_qubits + 1)
    idle_qubits = tuple(set(all_qubits) - set(active_qubits))
    
    # determining the constant voltage offset in case the actual voltage 
    # pulse is shifted anywhere in time
    v_offset = v_unit_shape(np.inf) # surely equal to offset at infinity
    delta_g_0 = delta_g_interp(v_offset)
    
    
    ## * Creating a ControlPulse Object *
    
    # Create a string of active qubits.
    actv_str = ""
    for q in active_qubits:
        if not isinstance(q, int) or q <= 0:
            raise Exception("active qubit parameter must be a tuple" +
                            "list, or array of positive integers")
        else:
            actv_str = actv_str + "_" + str(q) 
    
    # Format theta correctly
    if theta < 0: sign = "-"
    else: sign = ""
    theta_str = (sign + str(abs(theta))).zfill(3)
    
    if isinstance(axis, str):
        rot_pulse = ControlPulse(pulse_name='ROT{}{}{}'.format(axis, theta_str, actv_str), 
                                pulse_type='effective')
    else:
        n_x = rotation_axis[0]
        n_y = rotation_axis[1]
        n_z = rotation_axis[2]
        rot_pulse = ControlPulse(pulse_name='ROT({},{},{}){}{}'\
                    .format(n_x, n_y, n_z, theta, actv_str), pulse_type='effective')
    
    # setting all delta_g values to offset values for all qubits
    # setting all Voltage values of plunger gates
    # this will be updated for some qubits later on
    
    for i in all_qubits:
        rot_pulse.add_control_variable(var_name= 'delta_g_{ind}'.format(ind = i), 
                                    var_pulse=np.full(num_val, delta_g_0))
        rot_pulse.add_control_variable(var_name= 'V_{ind}'.format(ind = i), 
                                    var_pulse=np.full(num_val, v_offset))

    # values of normalized time and pulse
    tau = np.linspace(0, 1, num_val)
    v_pulse = v_unit_shape(tau)
    # Larmor frequency
    omega = e / m_e * B_0
    # Converting theta to radians and shifting it to the range (-pi, pi]
    theta_rad = np.deg2rad(theta) % (2 * np.pi)
    if theta_rad > np.pi:
        theta_rad -= 2 * np.pi
    # define alpha = theta / 2pi
    alpha = theta_rad / (2 * np.pi)
    
    # evaluating the integral of delta g pulse
    delta_g_pulse = delta_g_interp(v_unit_shape(tau))
    delta_g_int = np.trapz(delta_g_pulse, tau) - np.full(num_val, delta_g_0)
    
    # finding the pulse length and setting the correct time values for pulses
    T = 1.
    
    if axis == 'Z':
        # in this case, only the voltage on active qubits changes
        for i in active_qubits:
            rot_pulse.ctrl_pulses['delta_g_{ind}'.format(ind = i)] = delta_g_pulse
            rot_pulse.ctrl_pulses['V_{ind}'.format(ind = i)] = v_pulse
            
        # pulse duration
        T = 2 * theta_rad / (omega * np.abs(delta_g_int))
        
    elif axis in ['X', 'Y']:
        # determine the phase values during the pulse first
        phis = np.full(num_val, 0.)
        if axis.upper() == 'Y':
            phis = np.full(num_val, 90)
        if theta < 0:
            phis = phis + 180
            
        rot_pulse.add_control_variable("phi", phis)
        # Making the ESR pulse of the same shape as the g-factor pulse
        # shifted by delta_g_0
        # finding the coefficient first
        
        A = 1 / np.sqrt( (2 * np.pi / theta_rad) ** 2 - 1)
        
        # values of ESR field
        omega_capital = A * omega * np.abs(delta_g_pulse - delta_g_0) / 2
        b_rf = m_e / e * omega_capital 
        
        rot_pulse.add_control_variable("B_rf", b_rf)
        
        # unlike the case of ROTZ, the voltage here changes only on the 
        # idling qubits so that they return to their initial state
        # after the pulse
        for i in idle_qubits:
            rot_pulse.ctrl_pulses['delta_g_{ind}'.format(ind = i)] = delta_g_pulse
            rot_pulse.ctrl_pulses['V_{ind}'.format(ind = i)] = v_pulse
            
        # pulse duration
        alpha = theta_rad / (2 * np.pi)
        T = 4 * np.pi / (omega * np.abs(delta_g_int)) *  np.sqrt(1 - alpha ** 2)
    
    else:
        # *** ARBITRARY ROTATION ***
        #create variables for Vector Components to speed up run-time
        n_x = rotation_axis[0]
        n_y = rotation_axis[1]
        n_z = rotation_axis[2]
        
        # make sure that the rotation_axis is a Bloch Vector, if not, make it unitary
        norm = np.sqrt(n_x**2 + n_y**2 + n_z**2)
        if norm != 1:
            n_x = n_x/norm
            n_y = n_y/norm
            n_z = n_z/norm
        
        def find_shape(active = True):
            '''
            @author: Madi
            Calculates the shape function given that the qubit is
            resonant (active) or non-resonant (inactive) for T = 1

            Parameters
            ----------------
            active: boolean
                is True when the qubit is active, False when the qubit is inactive

            Returns
            ---------------
            List 
                contains all of the values of the shape
            '''
            delta_g_diff = np.array(delta_g_pulse - delta_g_0)
            if active:
                param = (2 * n_z * theta_rad)/omega
                return delta_g_diff / param
            
            else:
                param = (2/omega) * np.sqrt(4 * np.pi**2 - theta_rad**2 * 
                                                            (n_x**2 + n_y**2))
                return delta_g_diff / param
    
        # Find Shape
        if n_z == 1:
            S = find_shape(active = True) #get shape
        else:
            S = find_shape(active = False) #get shape
        
        # sign of pulse shape
        shape_sign = []
        for s in S:
            if s < 0:
                shape_sign.append(-1)
            else:
                shape_sign.append(1)
        
        #determine the phase values during the pulse -> list of phi's
        phis = np.zeros(num_val)
        for i, s in enumerate(shape_sign):
            if s > 0:
                phi = np.arctan2(n_y, n_x) * (180 / np.pi)
            else:
                phi = np.arctan2(n_y * -1, n_x * -1) * (180 / np.pi)
            phis[i] = phi
        rot_pulse.add_control_variable("phi", phis)# add phi's to the control variable
        
        
        #Calculate time of pulse for the rotation
        factor = ((4 * np.pi)/omega)
        if n_z == 0:
            T = factor * ((np.sqrt(1 - (alpha**2 * (n_x**2 + n_y**2)))) /
                                                            abs(delta_g_int))
        else:
            T = factor * ((alpha * abs(n_z)) / abs(delta_g_int))
        
        
        #Add ESR frequency into the Control Pulse
        omega_capital = []
        parameter = abs(theta_rad) * np.sqrt(n_x**2 + n_y**2) * m_e / e #saves calculation time
        for i, s in enumerate(S):
            omega_capital.append(s * parameter * shape_sign[i])
        rot_pulse.add_control_variable("B_rf", omega_capital)# add ESR frequency to the control variable
        
        
        
        # * Update the Control Pulse Object *
        # find dg+ - dg0 
        d_g_active = []
        parameter = (2 * n_z * theta_rad)/omega
        for s in S:
            d_g_active.append(s * parameter)
        
        #find dg- - dg0
        d_g_inactive= []
        parameter = (2/omega) * np.sqrt(4 * np.pi**2 - theta_rad**2 * 
                                                    (n_x**2 + n_y**2))
        for s in S:
            d_g_inactive.append(s * parameter)

        for i in active_qubits: 
            rot_pulse.ctrl_pulses['delta_g_{ind}'.format(ind = i)] = d_g_active
            if n_z == 1:
                rot_pulse.ctrl_pulses['V_{ind}'.format(ind = i)] = v_pulse

        for i in idle_qubits:
            rot_pulse.ctrl_pulses['delta_g_{ind}'.format(ind = i)] = d_g_inactive
            if n_z != 1:
                rot_pulse.ctrl_pulses['V_{ind}'.format(ind = i)] = v_pulse

    
    # finally, specify the correct time values for the pulse and returning it
    rot_pulse.add_control_variable('time', tau * T)
    
    return rot_pulse

    
    #TODO pandas recall how to add values
    

def create_ctlp_file(CntrlPulse, read = False, get_dir = False):
    '''
    @author: Madi

    Creates a .ctrlp file containing the information about a rotation gate
    from a ControlPulse object

    Parameters
    -------------
    CntrlPulse: ControlPulse
        Control Pulse Object 
    read: Bool
        If read = True, the function will print out the .ctrlp file, 
        if read = False (Default) the funciton will print nothing
    get_dir: Bool
        If get_dir = True, the function will return the directory as a 
        string, if get_dir = False (Default), the function will return nothing

    Returns
    -------------
    Str *conditional*
        Returns the directory of the file as a str if get_dir = True, otherwise,
        the function returns None

    None

    Effects
    -------------
    Creates a .ctrlp file containing all information to re-create the same ControlPulse 
    object

    Prints to screen *conditional* - prints the Control Pulse file to the screen if 
    read = True, otherwise, nothing is printed to the screen
    '''
    # retrieve all pieces of information
    name = CntrlPulse.name 
    ideal_gate = CntrlPulse.ideal_gate
    p_type = CntrlPulse.pulse_type
    p_length = CntrlPulse.length
    pulse_key = CntrlPulse.ctrl_names
    pulse_vals = list(CntrlPulse.ctrl_pulses.values())
    active_qubits = CntrlPulse.active
    
    # format Control Pulse data into writable strings
    key_str = ""
    vals_str = ""
    for i, key in enumerate(pulse_key):
        if i == len(pulse_key) -1:
            key_str = key_str + str(key).strip()
        else:
            key_str = key_str + str(key).strip() + "," 
    
    for j in range(len(pulse_vals[0])):
        vals_str_line = ""
        for i in range(len(pulse_key)):
            if i == (len(pulse_key) - 1):
                vals_str_line = vals_str_line + str(pulse_vals[i][j])
            else:
                vals_str_line = vals_str_line + str(pulse_vals[i][j]) + ", "
        if j == (len(pulse_vals[0]) - 1):
            vals_str = vals_str + vals_str_line
        else:
            vals_str = vals_str + vals_str_line + "\n"
    
    actv_str = ""
    if active_qubits != None:
        for i, q in enumerate(active_qubits):
            if i == (len(active_qubits) - 1):
                actv_str = actv_str + str(q)
            else:
                actv_str = actv_str + str(q) + ", "
    
    # Write the .ctrlp file
    ctrlp_file = open(name + '.ctrlp', 'w')
    ctrlp_file.writelines(["# {n}.ctrlp\n".format(n = name),
                           "Ideal gate: {ig}\n".format(ig = ideal_gate),
                           "Pulse type: {pt}\n".format(pt = p_type),
                           "Pulse length: {pl}\n".format(pl = p_length),
                           "Active qubits: {aq}\n".format(aq = actv_str),
                           "Control pulses:\n",
                            key_str + "\n",
                            vals_str])
    
    # close file to stop making any changes
    ctrlp_file.close()
    
    # read the file if necessary
    if read == True:
        directory = os.getcwd() + "/" + name + ".ctrlp"
        f = open(directory, 'r')
        print(f.read())
    
    # return the directory if necessary
    if get_dir == True:
        directory = os.getcwd() + "/" + name + ".ctrlp"
        return(directory)




def create_quantum_circuit(list_ctrlp, circuit_name, write_qcirc = False, print_dir = False):
    '''
    @author: Madi

    Takes directory to a .qcirc file, creates the necessary 
    ControlPulses for rotations around an axis, creates a dictionary 
    of these specified ControlPulse objects and returns a QuantumCircuit Object

    Parameters
    --------------
    list_ctrlp: Listof(Str(.ctrlp))
        A list that contains .ctrlp files where these files are specified by strings
    circuit_name: str
        The name that the user wishes to call the circuit
    write_qcirc: Bool
        If write_qcirc = True the function will write a qcirc file of the quantum circuit
        with the name "circuit_name".qcirc, else the function does nothing
    print_dir: Bool
        If write_qcirc = True and print_dir = True the function will print the directory 
        of the newly made .qcirc file, else the function does nothing

    Requires
    ----------
    * All ControlPulse objects act on the same number of qubits 

    * All .ctrlp files in list_ctrlp will path to the correct directory
        -> user must ensure that all directory satisfactions are met 

    Returns
    ---------
        QuantumCircuit object

    Effects
    ---------
    Writes File *conditional*
        writes a .qcirc file with the name "circuit_name".qcirc is write_qcirc = True
    Prints to Screen
        prints the directory of the newly made .qcirc file if print_dir = True and 
        write_qcirc = True
    '''
    pulse_dict = load_pulses(list_ctrlp) 
    
    file = open(list_ctrlp[0], 'r')
    L = file.readlines()
    for i, line in enumerate(L):
        if "control" in line.lower():
            ind = i + 1
            break
    cnt_dict = {'V':0}
    for s in L[ind]:
        if s == 'V':
            cnt_dict['V'] = cnt_dict['V'] + 1
    n_qubits = cnt_dict['V']
    
    file.close()
    
    ## ** Write .qcirc file if necessary **
    if write_qcirc == True:
        Gate_Str = ""
        
        # Create List of active qubits and List of qubit names, 
        # then make it into a writeable String
        for i in range(len(list_ctrlp)):
            f = open(list_ctrlp[i], 'r')
            L = f.readlines()
            actv_qubit_str = None
            name_str = None
            for param in L:
                if actv_qubit_str != None and name_str != None:
                    break
                elif "#" in param.lower(): #elif statement to format name correctly
                    name_str = param.split(".")[0].split("#")[1].strip()
                    if "ROT" in name_str:
                        name_str = name_str.replace("OT","")
                    num_count = 0
                    negative = ""
                    for s in name_str[name_str.index("_") + 1:]:
                        if s == "-":
                            negative = "-"
                        elif isinstance(int(s), int):
                            num_count = num_count + 1
                    if num_count == 1:
                        name_str = name_str.replace("_","{neg}00").format(neg = negative)
                    if num_count == 2:
                        name_str = name_str.replace("_","{neg}0").format(neg = negative)
                    else:
                        name_str = name_str.replace("_","{neg}").format(neg = negative)
                    
                elif "active" in param.lower():
                    actv_qubit_str = param.split(": ")[1].replace(",","")
            if i == (len(list_ctrlp) - 1):
                Gate_Str = Gate_Str + "{name} {active}".format(name = name_str, active = actv_qubit_str)
            else:
                Gate_Str = Gate_Str + "{name} {active}".format(name = name_str, active = actv_qubit_str)
        
        # create the new qcirc file
        qcirc_file = open(circuit_name + ".qcirc", 'w')
        qcirc_file.writelines(["Number of qubits: {nq}\n".format(nq = n_qubits), Gate_Str])
        qcirc_file.close()
        
        if print_dir == True:
            print(os.getcwd() + "/" + circuit_name + ".qcirc")

    
    ## ** Create Quantum Circuit Object with the newly made pulse_dict **
    return QuantumCircuit(circuit_name, n_qubits, pulse_dict)





























