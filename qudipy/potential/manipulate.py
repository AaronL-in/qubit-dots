'''
Function for the manipulation of potential data

@author: Madi Schuetze, Bohdan Khromets
'''

from scipy.signal import find_peaks
from scipy.optimize import minimize
import numpy as np
from qudipy.utils.constants import Constants


#TODO: add potential splitter from tutorials/Exchange_test.ipynb
                                            # in 'exchange' branch

def fit_quartic(gparams, y_slice=0, material='vacuum',
                                        return_params=True):
    '''
    Returns the potential data of best fit to a quartic potential within the 
    Heilter-London approximation. The data will be a two dot system  fitted 
    to a quartic well by minimizing ||U_data - U_fit|| by the method 
    of least squares.

    #TODO update to the case of many dots

    Parameters:
    -------------
    
    gparams: GridParameters Object
        Must contain 2D potential, and x and y coordinates as a meshgrid object


    Keyword Arguments:
    -------------------
    y_slice: Int, Function, optional 
        The y-value (or x-dependent function) of the 
        y_slice where the dots are located. Default is 0.

    #TODO find out where y-value being a function can be used

    material: String, optional
        The string defining the material that the dots are
        in. Default is 'vacuum'.

    return_params: Bool
        The option to return all of the fitting parameters
        as a dictionary with keys:
        * U_fit
        * xrange
        * yrange
        * dot_sep
        * e_field
        * omega0
        * x_centre
        * U0
        If False, only the potential landscape U_fit of best fit is returned

    Returns:
    --------------
    U_fit: 2D Arrayof(Float) 
    *or* result_dict: dictionary
        Contains Float values that map out the 1-D potential energy quartic fit
        of a two dot system.
    '''
    # Get x and y parameters
    x = gparams.x
    x_mesh = gparams.x_mesh

    y = gparams.y
    y_mesh = gparams.y_mesh
    
    U_data = gparams.potential

    length = len(U_data)

    # TODO: move the following to the GridParameters class ->
    # ->
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
    U_data_slice =  U_data[index]

    # Maxima
    maxima = find_peaks(U_data_slice)[0]
    # Minima
    minima = find_peaks(-1*U_data_slice)[0]
    
    # ** Determine Inital Guesses of Values **
    # Constants
    consts = Constants(material)
    e = consts.e # electron charge
    m = consts.m0 # mass of free electron
    hbar = consts.hbar # reduced plancks constant
    
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
    U0 = (E2 + E1)/2 # average lowest energy
    omega0 = 2/d * np.sqrt((1/m)*(2*Emax - E2 - E1)) # potential at x0
    
    # Electric field and radius of dot
    e_field = round(E2-E1, 3)/(e*2*d) # electric field
    
    # Create function for best fit
    def quartic_function(parameters):
        '''
        Creates a 2D Array of fit data

        Parameters:
        ------------
        parameters: 1D Arrayof(Float)
            Array containing values that define the well
            [e_field, omega0, d, x0, U0] - order important

        Returns:
        ------------
        U_fit: 2D Arrayof(Float)
            Contains the quartic potential of best fit
            depending on x and y 
        '''
        e_field, omega0, d, x0, U0 = parameters
        U_fit = (m * omega0**2)/2 *\
                (np.square(np.square(x_mesh - x0) - d**2) / (4*d**2)
                                                     + np.square(y_mesh)) +\
                e_field * e * x_mesh + U0
        return U_fit
    
    
    # Create function to minimize
    def minimize_func(parameters):
        '''
        Defines function to minimize

        Parameters:
        ------------
        parameters: 1D Arrayof(Float)
            Array containing values that define the well
            [e_field, omega0, d, x0, U0] - order important

        Returns:
        ------------
        Float: ||U_fit - U_data||
        '''
        U_fit = quartic_function(parameters)
        return np.linalg.norm(U_fit - U_data)
    
    # ** Find Optimal Fit **
    # Create initial guess
    guess = [e_field, omega0, d, x0, U0]
    # Find the minimum
    res = minimize(minimize_func, x0=guess, bounds=None, constraints=())

    # Return the optimized potential and parameters
    min_params = res.x
    min_e_field, min_omega0, min_d, min_x0, min_U0 = min_params
    
    U_best_fit = quartic_function(min_params)
    
    # Calculate relative error of fitting
    rel_err = np.linalg.norm(U_best_fit - U_data)/np.linalg.norm(U_best_fit)
    
    if return_params:
        result_dict = {'U_fit': U_best_fit, 'xrange': x, 'yrange': y,\
                'dot_sep': 2* min_d, 'e_field': min_e_field, 'omega0': min_omega0,\
                        'x_centre': min_x0, 'U0': U0, 'error': rel_err}
        return result_dict
    else:
        return U_best_fit

