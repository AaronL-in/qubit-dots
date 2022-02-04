
if __name__=='__main__':
    # for the case when the test is run from the "unit_tests" folder
    import sys, os
    up = os.path.abspath('..')
    sys.path.insert(0, up)

import numpy as np
from random import random

from qudipy.starkshift import starkshift




class TestStarkshiftObject:
    def test_temp(self):
        '''
        Verifies that the temperature parameter and change_temperature function
        works.
        '''
        T = self.temp
        self.change_temperature(T + 5)
        np.testing.assert_equal(self.temp, T + 5, "Temperatures Not equal, Test Failed", verbose=True)


    def test_temp_interp(self):
        '''
        Verifies that the correct interpolated g-factor value is returned 
        when different temperatures are specified in the object.
        '''
        self.change_temperature(0)
        delta_g = self.temp_g_factor('Si') #uses the temperature in the Class Object
        np.testing.assert_approx_equal(delta_g, 1.99875, significant = 6)

        self.change_temperature(100)
        delta_g = self.temp_g_factor('Si') #uses the temperature in the Class Object
        np.testing.assert_approx_equal(delta_g, 1.99865, significant = 6)


if __name__=='__main__':
    os.system('python -m pytest')