## *****************************************
##          Test Control
##          testing control module
## *****************************************

import os, sys
sys.path.append(os.path.dirname(os.getcwd()))

# import testing and data modules
import pytest
import numpy as np
from random import random

# import qudipy modules
from qudipy.control import shapes
from qudipy.starkshift import starkshift
from qudipy.control.voltage_pulses import balance_zeeman, rot


				## *** FROM SHAPES MODULE ***

def test_square_pulse():
	'''
	Checking that a square pulse with shifted boundaries is created 
	correctly
	'''
	time = range(10)
	square_shape = shapes.square( t_start=3, t_end=5, amp=2.5, offset=1)
	correct_pulse = np.array([1,1,1, 2.5,2.5,2.5, 1,1,1,1])
	np.testing.assert_array_equal(square_shape(time), correct_pulse)

def test_gauss_pulse_max():
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

def test_gauss_pulse_fwhm():
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

def test_shapes_comparison():
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

# *** define fixtures ***
@pytest.fixture(scope="module")
def shape_list():
	'''
	A fixture that returns a list of voltage pulse shapes 
	with set amplitude, offset, start time and end time
	values
	
	shapes(): None -> Listof(Array)
	'''
	amp =  40       
	offset =  10
	t_start = -25
	t_end = 5
	square_shape = shapes.square(t_start=t_start, 
									t_end=t_end, amp=amp, offset=offset)
	triangle_shape = shapes.triangle(t_start = t_start,
									t_end=t_end, amp=amp, offset = offset)
	wide_gauss = shapes.shifted_gauss(t_start = t_start,
										t_end=t_end, amp=amp, offset = offset, sigma = 0.4)
	narrow_gauss = shapes.shifted_gauss(t_start = t_start,
										t_end=t_end, amp=amp, offset = offset, sigma = 0.1)
	return [square_shape, triangle_shape, wide_gauss, narrow_gauss]

@pytest.fixture(scope="module")
def g_interp(v):
	'''
	A fixture which returns a list of interpolated
	values -> this is a dummy function for testing
			  purposes only!

	g_interp: Array/Listof(Float) -> Listof(Float)
	'''
	v = np.array(v)
	if np.shape(v) == ():
		return -1 * np.exp(0.01 * v)+1.9
	else:
		g = []
		for volt in v:
			g.append(-1 * np.exp(0.01 * volt)+1.9)
		return g

@pytest.fixture(scope="module")
def rot_save():
	'''
	Dictionary to save 'X','Y', and 'Z' rotations
	with their vector counterparts.

	rot_save: None -> Dict
	'''
	rots = {
			'X' : [1,0,0],
			'Y' : [0,1,0],
			'Z' : [0,0,1],
	}
	return rots


## *** run tests ***
def test_arbitrary_rot(shapes_list, g_interp, rot_save):
	B_0 = balance_zeeman(g_interp, square_shape, 1000)
	
	rot_list = rot_save.keys()

	for shape in shape_list: #test over multiple shapes
		for theta in [45, 90, 120, 180, 270, -45, -90, -120, -180, -270]: #test over multiple angles
			for qubits in [1,2,3,4]: #test over multiple qubits
				for r in rot_list:
					rot_pulse = rot(r, theta, qubits, [1], g_interp(0), shape, B_0, 100)
					arot_pulse = rot(rot_save[r], theta, qubits, [1], g_interp(0), shape, B_0, 100)
				
					# assert that times are the same
					rot_time = rot_pulse.ctrl_time
					arot_time = arot_pulse.ctrl_time
					assert rot_time == pytest.approx(arot_time, 1e-7), \
							"rotation time not equal"
				
					# assert phi is defined correctly
					assert shape(rot_pulse.ctrl_pulses['phi']) == 100,\
							"rot phi array not of correct length"
					assert shape(arot_pulse.ctrl_pulses['phi']) == 100, \
							"arot phi array not of correct length"
							
					# assert ESR frequency shape
					assert shape(rot_pulse.ctrl_pulses['B_rf']) == 100,\
							"rot ESR array not of correct length"
					assert shape(arot_pulse.ctrl_pulses['B_rf']) == 100,\
							"arot ESR array not of correct length"
					
					# assert delta_g_i shape
					assert shape(rot_pulse.ctrl_pulses['delta_g_{ind}'.format(ind = qubits)]) == 100,\
							"rot delta_g array not of correct shape"
					assert shape(arot_pulse.ctrl_pulses['delta_g_{ind}'.format(ind = qubits)]) == 100,\
							"arot delta_g array not of correct shape"
					
					# assert V_i shape
					assert shape(rot_pulse.ctrl_pulses['V_{ind}'.format(ind = qubits)]) == 100,\
							"rot V array not of correct shape"
					assert shape(arot_pulse.ctrl_pulses['V_{ind}'.format(ind = qubits)]) == 100,\
							"rot V array not of correct shape"
		
					