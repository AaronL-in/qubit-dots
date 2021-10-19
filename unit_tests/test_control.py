if __name__=='__main__':
    # for the case when the test is run from the "unit_tests" folder
    import sys, os
    up = os.path.abspath('..')
    sys.path.insert(0, up)

import numpy as np
from random import random

from qudipy.control import shapes

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

        

if __name__=='__main__':
    os.system('python -m pytest')