if __name__=='__main__':
    # for the case when the test is run from the "unit_tests" folder
    import sys, os
    up = os.path.abspath('..')
    sys.path.insert(0, up)

import numpy as np
from random import random

from qudipy.control import shapes
from qudipy.starkshift import starkshit

class TestPulseShapes:
    def test_square_pulse(self):
        """
            Checking that a square pulse with shifted boundaries is created 
            correctly
        """
        time = range(10)
        square_shape = shapes.square( t_start=3, t_end=5, amp=2.5, offset=1)
        correct_pulse = np.array([1,1,1, 2.5,2.5,2.5, 1,1,1,1])
        np.testing.assert_array_equal(square_shape(time), correct_pulse)
        

    def test_gauss_pulse_max(self):
        """
        testing that a Gaussian pulse reaches its maximum value, regardless
        of its duration, half-width or offset.
        """
        amp =  -10  ## deterministic, the maximum is always -10
        offset =  amp * random()
        t_start= (-10) * random()
        t_end = 10 * random()
        sigma = 2 * (t_end - t_start) * random()
        gauss_shape = shapes.shifted_gauss(t_start=t_start, t_end=t_end, 
                                            sigma=sigma, amp=amp, offset=offset)
        
        # finding value at the central point                                    
        t_center = (t_end + t_start) / 2
        np.testing.assert_allclose(gauss_shape(t_center), amp)
        

    def test_gauss_pulse_fwhm(self):
        """
            Checking that the full width at half maximum of a shifted Gaussian
            pulse is evaluated correctly. For small sigmas in relation to 
            pulse length, the value is almost equal to (amp + offset) / 2 
            because the vertical shift of the Gaussian is very small.
        """
        amp =  40       # the half-maximum is approximately (40-20)/2 = 10
        offset =  -20
        t_start = -10
        t_end = 15 
        sigma = 0.1 * (t_end - t_start)
        gauss_shape = shapes.shifted_gauss(t_start=t_start, t_end=t_end, 
                                            sigma=sigma, amp=amp, offset=offset)

        # checking if the half-maximum value is achieved
        fwhm = 2 * np.sqrt(2 * np.log(2)) * sigma
        times_at_half_max = np.array([(t_end + t_start - fwhm) / 2, 
                                (t_end + t_start + fwhm) / 2])
        vals_at_half_max = gauss_shape(times_at_half_max)

        # finding correct values taking into account that the pulse 
        # characteristic width can be actually larger than the pulse duration

        correct_vals = np.full((2,), (amp + offset) / 2)
        # correct_vals = np.full((2,), offset)
        # times_mask = (np.greater_equal(times_at_half_max, t_start) & 
        #                 np.less_equal(times_at_half_max, t_end))
        # correct_vals[times_mask] = np.full((2,), (amp + offset) / 2)[times_mask]
                                                
        np.testing.assert_array_almost_equal(vals_at_half_max, correct_vals, 
                                                                decimal=4)
    def test_shapes_comparison(self):
        """
            Comparing the pulses of different shapes and same parameters 
            otherwise. Verifies that square >= shifted_gauss >= triangle
            for large values of sigma (when FWHM is bigger than the pulse
            duration)
        """
        amp =  40       
        offset =  10
        t_start = -25
        t_end = 5
        sigma = 0.7 * (t_end - t_start)
        square_shape = shapes.square(t_start=t_start, t_end=t_end,
                                                 amp=amp, offset=offset)
        triangle_shape = shapes.triangle(t_start=t_start, t_end=t_end,
                                                 amp=amp, offset=offset)
        gauss_shape = shapes.shifted_gauss(t_start=t_start, t_end=t_end, 
                                            amp=amp, offset=offset, sigma=sigma)

        # the time interval covers both the proper pulse values and 
        # constant offset before/after the pulse
        times = np.linspace(-20, 20, 30)

        #comparing pulses
        comp_result = ((square_shape(times) >=  gauss_shape(times)) & 
                            (gauss_shape(times) >= triangle_shape(times))
                                        )
        np.testing.assert_equal(comp_result, True)


                ## *** FROM VOLTAGE PULSES MODULE ***
    def test_arbitrary_rotation_time(self):
        '''
        Verfifies that an arbitrary X, Y, and Z rotation return 
        the same values as the regular X, Y, and Z rotations

        Note: entering in a vector as the rotation axis instead of 
              the string 'X', 'Y', or 'Z' will result in the arbitrary
              rotation calculation being performed
        '''
        # Define a random interpolation of delta_g
        def g_interp(v):
            v = np.array(v)
            if np.shape(v) == ():
                return -1 * np.exp(0.01 * v)+1.9
            else:
                g = []
                for volt in v:
                    g.append(-1 * np.exp(0.01 * volt)+1.9)
                return g

        amp =  40       
        offset =  10
        t_start = -25
        t_end = 5
        sigma = 0.7 * (t_end - t_start)
        square_shape = shapes.square(t_start=t_start, 
                                        t_end=t_end, amp=amp, offset=offset)

        B0 = balance_zeeman(g_interp, square_shape, 1000)

        X_rot = rot('X', 90, 1, [1],g_interp(0), square_shape, B_0, 100)
        tx1 = X_rot.ctrl_time
        X_vec = rot([1,0,0] , 90, 1, [1],g_interp(0), square_shape, B_0, 100)
        tx2 = X_vec.ctrl_time
        
        Y_rot = rot('Y', 45, 2, [1],g_interp(0), square_shape, B_0, 100)
        ty1 = Y_rot.ctrl_time
        Y_vec = rot([0,1,0] , 45, 2, [1],g_interp(0), square_shape, B_0, 100)
        ty2 = Y_vec.ctrl_time
        
        Z_rot = rot('Z', 120, 1, [1],g_interp(0), square_shape, B_0, 100)
        tz1 = Z_rot.ctrl_time
        Z_vec = rot([0,0,1] , 120, 1, [1],g_interp(0), square_shape, B_0, 100)
        tz2 = Z_vec.ctrl_time

        correct_times = [tx1, ty1, tz1]
        check_times = [tx2, ty2, tz2]
        np.testing.assert_almost_equal(check_times, correct_times, 7)

    def test_retrieve_ctrlp_elements(self):
        '''
        Verifies that the rotation function defines and adds control pulse parameters 
        correctly
        '''
        check_list = []
        
        def g_interp(v):
            v = np.array(v)
            if np.shape(v) == ():
                return -1 * np.exp(0.01 * v)+1.9
            else:
                g = []
                for volt in v:
                    g.append(-1 * np.exp(0.01 * volt)+1.9)
                return g
            
        amp =  40       
        offset =  10
        t_start = -25
        t_end = 5
        sigma = 0.7 * (t_end - t_start)
        square_shape = shapes.square(t_start=t_start, 
                                        t_end=t_end, amp=amp, offset=offset)
        
        B0 = balance_zeeman(g_interp, square_shape, 1000)
        
        ZROT = rot('z', 120, 3, [3], g_interp, square_shape, B0, num_val=100) #note: 3 qubits
        #check that the phi variable is an array of num_val
        check_list.append(shape(ZROT.ctrl_pulses['phi']) == 100)
        #check that the ESR frequency is an array of num_val
        check_list.append(shape(ZROT.ctrl_pulses['B_rf']) == 100)
        #check that each of the delta_g_ind and V_ind lists are defined properly
        for i in range(1, 4): #range of qubits
            check_list.append(shape(ZROT.ctrl_pulses['delta_g_{ind}'.format(ind = i)]) == 100)
            check_list.append(shape(ZROT.ctrl_pulses['V_{ind}'.format(ind = i)]) == 100)

        XROT = rot('X', 45, 2, [1], g_interp, square_shape, B0, num_val=100) #note: 3 qubits
        #check that the phi variable is an array of num_val
        check_list.append(shape(XROT.ctrl_pulses['phi']) == 100)
        #check that the ESR frequency is an array of num_val
        check_list.append(shape(XROT.ctrl_pulses['B_rf']) == 100)
        #check that each of the delta_g_ind and V_ind lists are defined properly
        for i in range(1, 3): #range of qubits
            check_list.append(shape(XROT.ctrl_pulses['delta_g_{ind}'.format(ind = i)]) == 100)
            check_list.append(shape(XROT.ctrl_pulses['V_{ind}'.format(ind = i)]) == 100)

        AROT = rot([1,0,1], 90, 4, [2,4], g_interp, square_shape, B0, num_val=100) #note: 3 qubits
        #check that the phi variable is an array of num_val
        check_list.append(shape(AROT.ctrl_pulses['phi']) == 100)
        #check that the ESR frequency is an array of num_val
        check_list.append(shape(AROT.ctrl_pulses['B_rf']) == 100)
        #check that each of the delta_g_ind and V_ind lists are defined properly
        for i in range(1, 5): #range of qubits
            check_list.append(shape(AROT.ctrl_pulses['delta_g_{ind}'.format(ind = i)]) == 100)
            check_list.append(shape(AROT.ctrl_pulses['V_{ind}'.format(ind = i)]) == 100)

        correct = np.full(np.shape(check_list), True)
        np.testing.assert_array_equal(check_list, correct)

if __name__=='__main__':
    os.system('python -m pytest')