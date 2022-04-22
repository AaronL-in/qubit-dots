'''
File for approximating the given potential landscape for two dots.
--> Hund Mullikan Exchange 
@author: Madi Schuetze
'''
# ** Import Modules **
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.optimize import minimize
import numpy as np
from matplotlib import pyplot as plt
from qudipy.utils.constants import Constants
from qudipy.potential import GridParameters



def fit(U_data, gparams, y_slice=0, material='vacuum', return_params=False):
	'''
	Returns an array containing the Energy values 
	of best fit. The data will be a two dot system 
	fitted to a quartic well by minimizing
	||U_data - U_fit||by the method of least squares.

	Parameters:
	-------------
	U_data: 2DArrayof(Float)
		The 2-D array containing the energies of a 
		two dot potential well.

	gparams: GridParameters Object
		Defines the x and y coordinates as a meshgrid object

	Requires:
	-------------
	* The lengths of U_data, x, and y must all be the same

	Keyword Arguments:
	-------------------
	y_slice: Int, Function, optional 
		The y-value (or x-dependent function) of the 
		y_slice where the dots are located. Default is.

	material: String optional
		The string defining the material that the dots are
		in. Defualt is 'Si/SiO2'.

	return_params: Bool
		The option to return all of the information used
		as a dictionary with keys:
		* U_fit
		* xrange
		* yrange
		* dot_sep
		* e_field
		* omega0
		* x_centre
		* U0

	Returns:
	--------------
	U_fit: 2D Arrayof(Float) or dictionary
		Contains Float values that map out the 1-D 
		Energy quartic fit of a two dot system.
	'''
	# Get x and y parameters
	x = gparams.x
	x_mesh = gparams.x_mesh
	y = gparams.y
	y_mesh = gparams.y_mesh
	length = len(U_data)
	
	# Check for any errors in data
	if len(U_data[0]) != len(x):
		raise Exception('length of array of x-values does not match data.\n\
		x-length: {xl}\ndata_x_length: {dxl}'.format(xl=len(x), dxl=length))
	
	if length != len(y):
		raise Exception('length of array of y-values does not match data.\n\
		y-length: {yl}\ndata_x_length: {dyl}'.format(yl=len(y), dyl=length))
	
	# ** Find the maximum and minima **
	# Find U_data_values of y_slice used
	try:
		abs_val_array = np.abs(y - y_slice)
	except TypeError:
		print('y_slice must be the constant y-value at which the\
		 wells are lined up on. Currently, y_slice = {}'.format(y_slice))
	index = abs_val_array.argmin()
	U_data_slice = interp1d(x, U_data[index], kind='cubic')(x)

	# Maxima
	maxima = find_peaks(U_data_slice)[0]
	# Minima
	minima = find_peaks(-1*U_data_slice)[0]
	
	# ** Determine Inital Guesses of Values **
	# Constants
	e = Constants(material).e # electron charge
	m = Constants(material).m0 # mass of free electron
	hbar = Constants(material).hbar # reduced plancks constant
	
	# Indicies
	i0 = maxima[0]
	i1 = minima[0]
	i2 = minima[-1]
	
	# x-positions
	x0 = round(x[i0], 3) # x-value of local maximum
	x1 = x[i1] # x-value of first minimum
	x2 = x[i2] # x-value of second minimum
	d = (x2-x1)/2 # half dot seperation (initial guess)
	
	# Energies
	E1 = U_data_slice[i1] # min Energy of first well
	E2 = U_data_slice[i2] # min Energy of second well
	Emax = U_data_slice[i0] # Energy at local max
	U0 = (E2+E1)/2 # average lowest energy
	omega0 = 2/d * np.sqrt((1/m)*(2*Emax - E2 - E1)) # potential at x0
	
	# Electric field and radius of dot
	epsilon = round(E2-E1, 3)/(e*2*d) # electric field
	zeta = np.sqrt(hbar/(m * omega0)) # characteristic radius of dot
	
	# Create function for best fit
	def quartic_fit(parameters):
		'''
		Creates a 2D Array of fit data

		Parameters:
		------------
		parameters: 1D Arrayof(Float)
			Array containing values that define the well
			[epsilon, omega0, d, x0, U0] - order important

		Returns:
		------------
		U_fit: 2D Arrayof(Float)
			Contains the quartic potential of best fit
			depending on x and y 
		'''
		epsilon = parameters[0]
		omega0 = parameters[1]
		d = parameters[2]
		x0 = parameters[3]
		U0 = parameters[4]
		U_fit = (m * omega0**2)/2 *\
				(np.square(np.square(x_mesh - x0) - d**2) / (4*d**2) + np.square(y_mesh)) +\
				epsilon * e * x_mesh + U0
		return U_fit
	
	
	# Create function to minimize
	def minimize_func(parameters):
		'''
		Defines function to minimize

		Parameters:
		------------
		parameters: 1D Arrayof(Float)
			Array containing values that define the well
			[epsilon, omega0, d, x0, U0] - order important

		Returns:
		------------
		Float: ||U_fit - U_data||
		'''
		U_fit = quartic_fit(parameters)
		return np.linalg.norm(U_fit - U_data)
	
	# ** Find Optimal Fit **
	# Create initial guess
	guess = [epsilon, omega0, d, x0, U0]
	# Find the minimum
	res = minimize(minimize_func, x0=guess, bounds=None, constraints=())
	result = res.x
	
	# Return the optimized potential and parameters
	min_epsilon = result[0]
	min_omega0 = result[1]
	min_d = result[2]
	min_x0 = result[3]
	min_U0 = result[4]
	
	min_params = [min_epsilon, min_omega0, min_d, min_x0, min_U0]
	U_best_fit = quartic_fit(min_params)
	
	# Create Fitting Quantizer
	rel_err = np.linalg.norm(U_best_fit - U_data)/np.linalg.norm(U_best_fit)
	
	if return_params:
		result_dict = {'U_fit': U_best_fit, 'xrange': x, 'yrange': x,\
					'dot_sep': min_d, 'e_field': min_epsilon, 'omega0': min_omega0,\
					'x_centre': min_x0, 'U0': U0, 'error': rel_err}
		return result_dict
	else:
		return U_best_fit
