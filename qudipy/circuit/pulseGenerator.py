"""
Constant adiabaticity pulse generator

@author: aaranyaalex
"""

import numpy as np
import matplotlib.pyplot as plt
from qudipy.qutils.solvers import solve_schrodinger_eq
from scipy.optimize import fmin
from scipy.constants import hbar
import qudipy as qd
import qudipy.potential as pot


class PulseGen:

    def __init__(self, ctrl_pulse, pot_interp, adia_param):
        self.init_ctrl = ctrl_pulse
        self.n_ctrls = ctrl_pulse.n_ctrls
        self.pot_interp = pot_interp
        self.desired_ap = adia_param
        self.adia_pulse = []
        ctrl_pulse._generate_ctrl_interpolators()
        self.ctrl_interp = ctrl_pulse.ctrl_interps

    def optimizeAP(self):
        consts = self.pot_interp.constants
        X = self.pot_interp.x_coords
        times = np.linspace(0, self.init_ctrl.length, num=len(X))
        n_ctrls = self.n_ctrls

        # helper function for adiabaticity calculation
        def adiabatic(dt, psi, psi_ground, e_ens, desired_ap):
            xi = np.zeros(len(psi))
            for m in range(2*n_ctrls-2):
                xi += hbar * np.abs(np.divide(np.vdot(psi[:, m+1], dt * psi_ground.flatten()), (e_ens[0] - e_ens[m+1])))
            xi = xi - desired_ap
            return xi

        h = 1e-6  # in V
        coeff = [1, -8, 0, +8, -1]  # for 5 pt stencil
        delh = h*np.array([-2, -1, 0, 1, 2])  # control pulse voltage shifts
        dpsi_dv = 0
        volt_vec = np.zeros((len(times), n_ctrls))

        for idx, ctrl in enumerate(self.init_ctrl.ctrl_names):
            # array of interpolated control pulses (no deltaV shift)
            volt_vec[:, idx] = self.ctrl_interp[ctrl](times)

        for i in range(n_ctrls):
            for s in range(5):
                if s == 2:
                    int_pot = self.pot_interp(volt_vec)
                    gparams = pot.GridParameters(X, potential=int_pot)
                    # center of 5 point stencil, store wf and energies for all accessible states
                    ce_ens, ce_vecs = solve_schrodinger_eq(consts, gparams, n_sols=2*n_ctrls-1)
                else:
                    # values of stencil, only requires ground state
                    int_pot = self.pot_interp(volt_vec+delh[s])  # interpolate potential based on small shift of ctrl
                    gparams = pot.GridParameters(X, potential=int_pot)
                    _, e_vecs = solve_schrodinger_eq(consts, gparams, n_sols=1)
                    dpsi_dv += e_vecs * coeff[s]

            # approximate the derivative of ground state
            dpsi_dv = dpsi_dv/(12*h)
            dvi_dt = np.gradient(volt_vec, edge_order=2)
            # find optimal dv/dt to achieve adiabatic parameter, set current dv/dt as initial guess
            grad = fmin(adiabatic, dvi_dt, args=(ce_vecs, dpsi_dv, ce_ens, self.desired_ap))
            # modify the time indices
            opt_times = times
            opt_times[1:] = times[:-1] + np.divide(np.diff(volt_vec[:, i]), grad[1:])

            self.adia_pulse[:, i+n_ctrls] = opt_times
            self.adia_pulse[:, i] = volt_vec[:, i]


if __name__ == '__main__':
    pot_dir = 'C:/Users/aaran/Documents/3B/URA/QuDiMy/tutorials/QuDiPy tutorial data/Pre-processed potentials/'

    # Specify the control voltage names (C#NAME as mentioned above)
    ctrl_names = ['V1', 'V2', 'V3', 'V4', 'V5']

    # Specify the control voltage values you wish to load.
    # The cartesian product of all these supplied voltages will be loaded and MUST exist in the directory.
    V1 = [0.1]
    V2 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
    V3 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
    V4 = [0.2, 0.22, 0.24, 0.26, 0.27, 0.28]
    V5 = [0.1]
    # Add all voltage values to a list
    ctrl_vals = [V1, V2, V3, V4, V5]

    # Now load the potentials.
    # load_files returns a dictionary of all the information loaded
    loaded_data = pot.load_potentials(ctrl_vals, ctrl_names, f_type='pot',
                                      f_dir=pot_dir, f_pot_units="eV",
                                      f_dis_units="nm", trim_x=[-120E-9, 120E-9])

    # Now building the interpolator object is trivial
    # Note the use of y-slice here which builds a 1D potential
    # interpolator at y=0 for all potentials.
    pot_interp = pot.build_interpolator(loaded_data,
                                        constants=qd.Constants("Si/SiO2"),
                                        y_slice=0)
    # Define some min and max voltage values
    min_v = 0.2
    max_v = 0.278

    # Specify the first point.
    pt1 = [max_v, min_v, min_v]
    res_vv = pot_interp.find_resonant_tc(pt1, 'V3')

    # Find the voltage configuration close to the resonant tunnel coupling point.
    almost_res_vv = res_vv - 0.035 * (max_v - min_v)
    pt2 = pt1.copy()
    pt2[1] = almost_res_vv

    # Find the voltage configuration right at the resonant tunnel coupling point
    pt3 = pt2.copy()
    pt3[1] = res_vv
    # Sweep V1 to the close to resonant tunnel coupling voltage configuration
    pt4 = pt3.copy()
    pt4[0] = almost_res_vv * 1.01

    # Sweep V1 to the minimum voltage to full localize the dot underneath V3.
    pt5 = pt4.copy()
    pt5[0] = min_v
    # Find resonant tunnel coupling point w.r.t. V4
    res_vv = pot_interp.find_resonant_tc(pt5, 'V4')

    # Sweep V4 to almost the resonant tunnel coupling point
    pt6 = pt5.copy()
    pt6[2] = almost_res_vv * 1.01

    # Sweep V4 to the resonant tunnel coupling point
    pt7 = pt6.copy()
    pt7[2] = res_vv

    # Sweep V3 to almost the resonant tunnel coupling point
    pt8 = pt7.copy()
    pt8[1] = almost_res_vv

    # Sweep V3 to the minimum voltage and fully localize
    # the electron underneath V4
    pt9 = pt8.copy()
    pt9[1] = min_v
    shuttle_pulse = np.array([pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9])

    # Specify the total overall length of the control pulse in seconds
    pulse_length = 10E-12  # 10 ps
    # Initialize a ControlPulse with name 'shuttle_exp' and
    # 'experimental' control variables.
    shut_pulse = qd.circuit.ControlPulse('shuttle_exp', 'experimental',
                                         pulse_length=pulse_length)

    # Add each voltage control variable now to the ControlPulse object
    shut_pulse.add_control_variable('V2', shuttle_pulse[:, 0])
    shut_pulse.add_control_variable('V3', shuttle_pulse[:, 1])
    shut_pulse.add_control_variable('V4', shuttle_pulse[:, 2])
    # Specify a time array to correspond to each voltage configuration in the pulse
    ctrl_time = pulse_length * np.array([0, 1 / 20, 1 / 4, 1 / 2 - 1 / 20, 1 / 2, 1 / 2 + 1 / 20, 3 / 4, 19 / 20, 1])
    shut_pulse.add_control_variable('Time', ctrl_time)
    shut_pulse.set_pulse_length(10e-12)
    apulse = PulseGen(shut_pulse, pot_interp, 0.02)
    apulse.optimizeAP()