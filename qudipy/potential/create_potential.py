'''
Class for potentials - issue #40

@author Madi
'''

import numpy as np
from  qudipy.potential import grid_params
import qudipy as qd 

class Potential:
	def __init__(self, omegas, dot_sep, material, hor_comp = 10000):
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
		if ctrl_num % 2 == 0:
			mid = int(ctrl_num/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.delete(np.arange(lbound, ubound), mid)
		
		else:
			mid = int((ctrl_num - 1)/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.arange(lbound, ubound)
		
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
		
		# create offset list
		if ctrl_num % 2 == 0:
			mid = int(ctrl_num/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.delete(np.arange(lbound, ubound), mid)
			
		else:
			mid = int((ctrl_num - 1)/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.arange(lbound, ubound)
			
		# Create list of wells
		well_list = []
		for i, off in enumerate(offset):
			v_i = ctrl_vals[i]
			mu = v_i * self.consts.e
			omega = self.omegas[i]
			well = 1/2 * omega * self.consts.e *\
					(np.power((scale * x_pot + off * sep), 4) + np.power(scale * y_pot,4)) - mu
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
		lenx = round(len(x_pot)/6)
		leny = round(len(y_pot)/6)
		
		# create offset list - even
		if ctrl_num % 2 == 0:
			mid = int(ctrl_num/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.delete(np.arange(lbound, ubound), mid)
		
		#create offset list - odd
		else:
			mid = int((ctrl_num - 1)/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.arange(lbound, ubound)
		
		# Find Spacing Dictionary
		spacing_dict = {}
		for off in offset:
			lbound = (off - 1/4) * dot_sep 
			ubound = (off + 1/4) * dot_sep
			spacing_dict[(lbound, ubound)] = None
		
		# Fill in spacing dict with position numbers
		# of where the well is non-zero
		x = x_pot[0] #sample x-slice
		for s in list(spacing_dict.keys()):
			ind_list = []
			for i in range(len(x)):
				if s[0] <= x[i] <= s[1]:
					ind_list = ind_list + [i]
			spacing_dict[s] = (min(ind_list), max(ind_list))
				
		# Create list of wells
		well_list = []
		x_pot_new = x_pot * 0 
		y_pot_new = y_pot * 0
		for ind in range(len(x_pot)):
			for i, s_ind in enumerate(list(spacing_dict.values())):
				v_i = ctrl_vals[i]
				mu = v_i * self.consts.e
				omega = self.omegas[i]
				# create x and y values array
				xval = x_pot[ind][0] * 1/2 * self.consts.me * omega**2 - v_i
				yval = y_pot[ind][0] * 1/2 * self.consts.me * omega**2 - v_i
				for j in range(s_ind[0], s_ind[1] + 1):
					x_pot_new[ind][j] = xval
					y_pot_new[ind][j] = yval

			well = x_pot_new + y_pot_new
			well_list = well_list + [well]
			
		return_well = well_list[0]
		for w in well_list:
			return_well = np.minimum(return_well, w)
		
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
		lenx = round(len(x_pot)/6)
		leny = round(len(y_pot)/6)
		
		# create offset list - even
		if ctrl_num % 2 == 0:
			mid = int(ctrl_num/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.delete(np.arange(lbound, ubound), mid)
			
		#create offset list - odd
		else:
			mid = int((ctrl_num - 1)/2)
			lbound = -1 * mid
			ubound = mid + 1
			offset = np.arange(lbound, ubound)
			
		# Find Spacing Dictionary
		spacing_dict = {}
		for off in offset:
			lbound = (off - 1/4) * dot_sep 
			ubound = (off + 1/4) * dot_sep
			spacing_dict[(lbound, ubound)] = None
			
		# Fill in spacing dict with position numbers
		# of where the well is non-zero
		x = x_pot[0] #sample x-slice
		for s in list(spacing_dict.keys()):
			ind_list = []
			for i in range(len(x)):
				if s[0] <= x[i] <= s[1]:
					ind_list = ind_list + [i]
			spacing_dict[s] = (min(ind_list), max(ind_list))
			
		# Create list of wells
		well_list = []
		x_pot_new = x_pot * 0 
		y_pot_new = y_pot * 0
		for ind in range(len(x_pot)):
			for i, s_ind in enumerate(list(spacing_dict.values())):
				v_i = ctrl_vals[i]
				mu = v_i * self.consts.e
				omega = self.omegas[i]
				# create x and y values array
				if i == 0:
					range_vals = []
					for j in range(s_ind[0], s_ind[1] + 1):
						x_pot_new[ind][j] = x_pot[ind][j] * self.consts.me * v_i * omega**2 - mu
						y_pot_new[ind][j] = y_pot[ind][j] * self.consts.me * v_i * omega**2 - mu
						range_vals = range_vals + [j]
				else:
					for k, j in enumerate(range(s_ind[0], s_ind[1] + 1)):
						r = range_vals[k]
						x_pot_new[ind][j] = x_pot_new[ind][r]
						y_pot_new[ind][j] = y_pot_new[ind][r]
				
			well = x_pot_new + y_pot_new
			well_list = well_list + [well]
			
		return_well = well_list[0]
		for w in well_list:
			return_well = np.minimum(return_well, w)
			
		return return_well
