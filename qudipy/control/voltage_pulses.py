'''
    Module that contains methods that design "simple" one- and two-qubit gates.
    Will return control pulse objects eventually, but now it is used only to 
    find pulse duration or amplitude
'''

#TODO:
# search function for delta g = g_0 for all qubits
# 

import numpy as np
import pandas as pd
from qudipy.control import shapes
from qudipy.circuit.control_pulse import ControlPulse
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
    v_offset: function
        Voltage offset when no pulse is applied.
    f_rf: float
        ESR frequency.

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



def rot(axis, theta, n_qubits, active_qubits, 
                delta_g_interp, v_unit_shape, B_0, num_val=100):
    '''
    Chooses the optimal duration for a ROT(\theta) pulse of the specified pulse 
    shape, where the argument 'axis' defines rotation. 
    ***For now*** the function assumes for the sake of simplicity that
    the delta g interpolators are the same for each qubit

    Parameters
    -----------------
    axis: string
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
    # build a tuple of positions of inactive qubits
    all_qubits = range(1, n_qubits + 1)
    idle_qubits = tuple(set(all_qubits) - set(active_qubits))

    # determining the constant voltage offset in case the actual voltage 
    # pulse is shifted anywhere in time
    v_offset = v_unit_shape(np.inf) # surely equal to offset at infinity
    delta_g_0 = delta_g_interp(v_offset)

    # creating a control pulse object 

    rot_pulse = ControlPulse(pulse_name='ROT{}_{}'.format(axis.upper(), theta), 
                                pulse_type='effective')

    # setting all delta_g values to offset values for all qubits
    # this will be updated for some qubits later on

    for i in all_qubits:
        rot_pulse.add_control_variable(var_name=f'delta_g_{i}', 
                                    var_pulse=np.full(num_val, delta_g_0))
        rot_pulse.add_control_variable(var_name=f'V_{i}', 
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

    # evaluating the integral of delta g pulse
    delta_g_pulse = delta_g_interp(v_unit_shape(tau))
    delta_g_int = np.trapz(delta_g_pulse, tau) - delta_g_0

    # finding the pulse length and setting the correct time values for pulses
    T = 1.

    if axis.upper()=='Z':
        # in this case, only the voltage on active qubits changes
        for i in active_qubits:
            rot_pulse.ctrl_pulses[f'delta_g_{i}'] = delta_g_pulse
            rot_pulse.ctrl_pulses[f'V_{i}'] = v_pulse

        # pulse duration
        T = 2 * theta_rad / (omega * delta_g_int)

    if axis.upper() in ('X', 'Y'):
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
            rot_pulse.ctrl_pulses[f'delta_g_{i}'] = delta_g_pulse
            rot_pulse.ctrl_pulses[f'V_{i}'] = v_pulse

        # pulse duration
        alpha = theta_rad / (2 * np.pi)
        T = 4 * np.pi / (omega * delta_g_int) *  np.sqrt(1 - alpha ** 2)
        

    # finally, specifying the correct time values for the pulse and returning it
    rot_pulse.add_control_variable('time', tau * T)

    return rot_pulse
    


    #TODO pandas recall how to add values






