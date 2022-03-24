## *****************************************
##          Test Starkshift
##          testing starkshift module
## *****************************************

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))


import numpy as np
from random import random
import pytest

from qudipy.starkshift import starkshift




def test_temp():
    '''
    Verifies that the temperature parameter and change_temperature function
    works.
    '''
    T = self.temp
    self.change_temperature(T + 5)
    assert self.temp == T + 5, "Temperatures Not equal, Test Failed"


def test_temp_interp():
    '''
    Verifies that the correct interpolated g-factor value is returned 
    when different temperatures are specified in the object.
    '''
    self.change_temperature(0)
    delta_g = self.temp_g_factor('Si') #uses the temperature in the Class Object
    assert delta_g == pytest.approx(1.99875, 1e-6)

    self.change_temperature(100)
    delta_g = self.temp_g_factor('Si') #uses the temperature in the Class Object
    assert delta_g == pytest.approx(1.99865, 1e-6)

    