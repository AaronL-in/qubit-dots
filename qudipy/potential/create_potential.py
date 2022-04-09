'''
Class for potentials - issue #40

@author Madi
'''

import numpy as np
from  qudipy.potential import grid_params
import qudipy as qd 

class Potential:
	def __init__(self, omegas, dot_sep, material, hor_comp = 10000, bias = 0, limit = 1/4):
		'''
		Initialize Potential object

		Parameters
		-------------
		omega: Listof(Float)
			frequency of harmonic oscillators

		dot_sep: Float
			seperation of dots

		material: String
			determines that constants that we use. Must
			be consistent with Constants class in 
			qudipy/utils/constants.py 

		hor_comp: Int
			The horizontal compression of the well for 
			the quartic well.
			Default = 10000

		bias: Anyof(Int, Float)
			The constant value that the potential takes 
			for a square or triangular well.
			Default = 0

		limit: Float
			A half the fraction of the dot seperation that the well 
			will span
		
		Requires
		------------
		* len(omegas) must be equal to len(ctrl_vals)
		  the length of the number of voltages specified
		  in the analytic_potential function
		
		Returns
		-------------
		None
		
		Note
		-------------
		All Potential objects are eligible to be used in the
		analytic_interpolator function
		'''
		self.omegas = omegas
		self.dot_sep = dot_sep
		self.material = material
		self.consts = qd.Constants(material)
		self.hor_scale = hor_comp
		self.bias = bias
		self.limit = limit
	
	
	def harmonic_potential(self, ctrl_vals, gparams):
		'''
		Creates an array of values that produces a 
		harmonic potential well

		Parameters
		-------------
		ctrl_vals: List
			List of voltages on voltage gates

		gparams: GridParameters object
			specifies the x and y coordinates

		Returns
		-------------
		Arrayof(Arrayof(Float)) 
		'''
		# Define variables needed for later
		ctrl_num = len(ctrl_vals)
		x_pot = gparams.x_mesh
		y_pot = gparams.y_mesh
		sep = self.dot_sep
		
		# create offset list
		offset = offset_list(ctrl_num)
		
		# Create list of wells
		well_list = []
		for i, off in enumerate(offset):
			v_i = ctrl_vals[i]
			mu = v_i * self.consts.e
			omega = self.omegas[i]
			well = 1/2 * self.consts.me * omega**2 *\
					(np.square(x_pot + off * sep) + np.square(y_pot)) - mu
			
			well_list = well_list + [well]
		
		min = well_list[0]
		for w in well_list:
			min = np.minimum(min, w)
			
		return min

	def quartic_potential(self, ctrl_vals, gparams):
		'''
		Creates an array of values that produces a 
		quartic potential well

		Parameters
		-------------
		ctrl_vals: List
			List of voltages on voltage gates

		gparams: GridParameters object
			specifies the x and y coordinates

		Returns
		-------------
		Arrayof(Arrayof(Float)) 
		'''
		# Define variables needed for later
		ctrl_num = len(ctrl_vals)
		x_pot = gparams.x_mesh
		y_pot = gparams.y_mesh
		sep = self.dot_sep
		scale = self.hor_scale
		
		# Create offset list
		offset = offset_list(ctrl_num)
			
		# Create list of wells
		well_list = []
		for i, off in enumerate(offset):
			v_i = ctrl_vals[i]
			mu =  10 * v_i * self.consts.e
			omega = self.omegas[i]
			well = 1/2 * omega * self.consts.e *\
					(np.power((scale * (x_pot + off * sep)), 4) + np.power(scale * y_pot,4)) - mu
			well_list = well_list + [well]
			
		min = well_list[0]
		for w in well_list:
			min = np.minimum(min, w)
			
		return min 

	
	def square_potential(self, ctrl_vals, gparams):
		'''
		Creates an array of values that produces a 
		square potential well

		Parameters
		-------------
		ctrl_vals: List
			List of voltages on voltage gates

		gparams: GridParameters object
			specifies the x and y coordinates

		bias: Anyof(Int, Float)
			The constant voltage value that the well 
			normally takes when it is not on a dip.

		Returns
		-------------
		Arrayof(Arrayof(Float)) 

		Requires:
		-------------
		gparams.x_mesh = gparams.y_mesh
		'''
		# Define variables needed for later
		ctrl_num = len(ctrl_vals)
		x_pot = gparams.x_mesh
		y_pot = gparams.y_mesh
		dot_sep = self.dot_sep
		bias = self.bias
		limit = self.limit
		
		# Create offset list
		offset = offset_list(ctrl_num)
		
		# Create Well Array
		def square(mesh, offset, limit, bias, dot_sep):
			'''
			Creates an array of mesh objects representing a 1-D square well

			Parameters
			-------------
			mesh: ndarray (mesh grid object)
				The x or y mesh object that defines the positions
			offset: Array
				The array containing the offset positions of each well
			limit: Float
				The a half the fraction of the dot seperation that the well 
				will span
			bias: Anyof(Int, Float)
				The initial voltage in Volts
			dot_sep: Anyof(Int, Float)
				The dot seperation between wells

			Returns
			-------------
			Arrayof(mesh object):
				Returns an array of mesh objects Arrayof(ndarray) 
				of a square well.

			Requires
			-------------
			* 2*limit < dot seperation
			'''
			omegas = self.omegas 
			well = bias * np.ones(np.shape(mesh))
			well_list = []
			for i, off in enumerate(offset):
				v_i = ctrl_vals[i]
				omega = omegas[i]
				amp = 1/2 * self.consts.me * omega**2 - v_i
				mask =  ((mesh >= (off - limit)*dot_sep) & (mesh <= (off + limit)*dot_sep))
				well[mask] = amp
				well_list = well_list + [well]
				
			return np.array(well_list)

		# Create x_well array and y_well array
		x_wells = square(x_pot, offset, limit, bias, dot_sep)
		y_wells = square(y_pot, [0], limit, bias, dot_sep)

		# Add arrays and take the minimum
		well_array = x_wells + y_wells
		
		return_well = well_array[0]
		for w in well_array:
			return_well = np.minimum(return_well, w)
		
		mask = (return_well > -0.3)
		return_well[mask] = bias
		
		# Return the final well
		return return_well 
	
	
	def triangle_potential(self, ctrl_vals, gparams):
		'''
		Creates an array of lists containing the values that
		a triangular potential well would contain

		Parameters
		-------------
		ctrl_vals: List
			List of voltages on voltage gates
		
		gparams: GridParameters object
			specifies the x and y coordinates
		
		Returns
		-------------
		Arrayof(Arrayof(Float)) 
		'''
		# Define variables needed for later
		ctrl_num = len(ctrl_vals)
		x_pot = gparams.x_mesh
		y_pot = gparams.y_mesh
		dot_sep = self.dot_sep
		bias = self.bias
		limit = self.limit
		
		# Create offset list
		offset = offset_list(ctrl_num)
		
		# Create Well Array
		def triangle(mesh, offset, limit, bias, dot_sep):
			'''
			Creates an array of mesh objects representing a 1-D square well

			Parameters
			-------------
			mesh: ndarray (mesh grid object)
				The x or y mesh object that defines the positions
			offset: Array
				The array containing the offset positions of each well
			limit: Float
				The a half the fraction of the dot seperation that the well 
				will span
			bias: Anyof(Int, Float)
				The initial voltage in Volts
			dot_sep: Anyof(Int, Float)
				The dot seperation between wells

			Returns
			-------------
			Arrayof(mesh object):
				Returns an array of mesh objects Arrayof(ndarray) 
				of a square well.

			Requires
			-------------
			* 2*limit < dot seperation
			'''
			omegas = self.omegas 
			well = bias * np.ones(np.shape(mesh))
			well_list = []
			
			for i, off in enumerate(offset):
				v_i = ctrl_vals[i]
				mask =  ((mesh >= (off - limit)*dot_sep) & (mesh <= (off + limit)*dot_sep))
				# Save the potential values to use each time
				if i == 0:
					save_vals = mesh[mask]
				well[mask] = save_vals * v_i
				well_list = well_list + [well]
			return np.array(well_list)
		
		# Create x_well array and y_well array
		x_wells = triangle(x_pot, offset, limit, bias, dot_sep)
		y_wells = triangle(y_pot, [0], limit, bias, dot_sep)
		
		# Add arrays and take the minimum
		well_array = x_wells + y_wells
		
		return_well = well_array[0]
		for w in well_array:
			return_well = np.minimum(return_well, w)
		
		# Return the final well
		return return_well


def offset_list(n_wells):
	'''
	Creates offset list for Potential Class
	functions

	Parameters
	-------------
	ctrl_num: Int
		The number of wells

	Returns
	-------------
	Returns the integer offset list used
	in the Potential Class functions
	'''
	# create offset list - even
	if n_wells % 2 == 0:
		mid = int(n_wells/2)
		lbound = -1 * mid
		ubound = mid + 1
		offset = np.delete(np.arange(lbound, ubound), mid)
		
	#create offset list - odd
	else:
		mid = int((n_wells - 1)/2)
		lbound = -1 * mid
		ubound = mid + 1
		offset = np.arange(lbound, ubound)
	
	return offset