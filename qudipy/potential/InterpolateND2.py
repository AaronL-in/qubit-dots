import numpy as np
from itertools import product

# import qudipy as qd
# from qudipy.potential import GridParameters
# # from qudipy.potential import InterpolateND
# from scipy.interpolate import RegularGridInterpolator


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import interpn

class InterpolateND2:

    def __init__(self, data, interp_vals):

        # 3D interpolation 

        n_gates = len(data['voltages'])

        # Set up grid for plotting
        X, Y= np.meshgrid(data['voltages'][0], data['voltages'][1])

        # mesh = np.mgrid[min(data['voltages'][0]):len(data['voltages'][0])+1:max(data['voltages'][0]),
        #     min(data['voltages'][1]):len(data['voltages'][1])+1:max(data['voltages'][1])]
        # mesh = np.mgrid[data['voltages'][0], data['voltages'][1]]

        # Note the following two lines that are used to set up the
        # interpolation points as a 10x2 array!
        interp_mesh = np.array(np.meshgrid(interp_vals[0], interp_vals[1]))
        interp_points = np.rollaxis(interp_mesh, 0, 3).reshape((5,2))

        # Perform the interpolation
        interp_arr = interpn((data['voltages'][0], data['voltages'][1]), data['response'][0], interp_points)


        # Plot the values as a surface plot to depict
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.plot_surface(X, Y, data['response'][0], rstride=1, cstride=1, cmap=cm.jet,
                            linewidth=0, alpha=0.8)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Plot the result
        # xx, yy, zz = np.meshgrid(interp_vals[0], interp_vals[1], interp_vals[2])
        xx, yy = np.meshgrid(interp_vals[0], interp_vals[1])
        # ax.scatter(xs = xx, ys = yy, zs = zz, c= interp_arr[0,:], s=20)
        # ax.scatter(xs = xx, ys = yy, c= interp_arr[4,:], s=20)
        ax.scatter(interp_x * np.ones(interp_y.shape), interp_y, interp_arr, s=20,
           c='k', depthshade=False)
                
        ax.set_xlabel('v1')
        ax.set_ylabel('v2')
        ax.set_zlabel('w1')
        plt.show()

        someting = 10

if __name__ == "__main__":

    data = {'voltages': {}, 'response': {}}
    scale = 5

    # Set up grid and array of values
    t1 = np.arange(scale)
    t2 = np.arange(scale)
    t3 = np.arange(scale)
    # x1 = np.linspace(-0.05, 1.5, 10)
    # x2 = np.linspace(0.2, 0.5, 10)

    data['response'][0] = t1 + t2[:, np.newaxis]

    x1 = np.linspace(-0.05, 1.5, scale)
    x2 = np.linspace(0.2, 0.5, scale)
    # x1 = t1
    # x2 = t2
    
    data['voltages'][0] = x1
    data['voltages'][1] = x2

    interp_x = 0.25           # Only one value on the x1-axis
    interp_y = np.linspace(0.3, 0.45, scale) # A range of values on the x2-axis
    interp_z = np.linspace(0.1, 0.7, scale) 

    # interp_x = 5           # Only one value on the x1-axis
    # interp_y = np.linspace(3.5, 7.5, 10) # A range of values on the x2-axis

    interp_vals = [interp_x, interp_y]

    interp_obj = InterpolateND2(data, interp_vals)
