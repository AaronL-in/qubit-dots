from logging import raiseExceptions
import numpy as np
from numpy import ma
from itertools import product
from itertools import compress

import matplotlib.pyplot as plt
from matplotlib import cm

from scipy.interpolate import interpn

class InterpolateND:

    def __init__(self, data, *coord):

        # store corrdinate information
        self.coord = coord
        self.response = data

        # remove sigleton dimensions
        self.mask = self.singleton()
        self.new_vecs = tuple(compress(self.coord, self.mask))

        print(f'{sum(np.invert(self.mask))} singleton dimensions for {len(self.mask)} dimensional data set')

        # define mesh grid for raw data
        self.M = np.meshgrid(*self.new_vecs)
        

    # Perform the interpolation
    def __call__(self, *interp_vals):
        
        # store interpolation vectors for all dimensions
        self.interp_vals = interp_vals

        # identify any singleton dimensions
        no_singleton = []
        for idx, dim in enumerate(np.shape(self.response)):
            
            # generate index mask to extract subset of data array with no
            # singleton dimenstions
            if self.mask[idx] == False:
                no_singleton.append([0])
            elif self.mask[idx] == True:
                no_singleton.append(list(range(dim)))
            # if self.mask[idx] == True:
            #     no_singleton.append(idx)

        # print(no_singleton)

        # print(np.ix_(*no_singleton))

        self.new_dat = np.squeeze(self.response[np.ix_(*no_singleton)])

        # print(np.array_equal(self.response[:,0,:,:], new_dat))

        # new_dat = self.response[no_singleton]
        # l = [0,2,3]
        # new_dat = self.response[l]
        
        # new_dat = tuple(compress(self.response, self.mask))

        # self.new_dat = self.response[:,0,:,:]



        # new_dat = self.response[self.mask]

        self.new_interp = tuple(compress(self.interp_vals, self.mask))

        # define mesh grid for interpolation

        self.Mi = np.meshgrid(*self.new_interp)
        interp_mesh = np.array(self.Mi).T

        self.interp_arr = interpn(tuple(self.new_vecs),
         self.new_dat, 
         interp_mesh)

        return self.interp_arr
        
    def singleton(self):

        mask = np.zeros((len(self.coord)))
        for idx, dim in enumerate(self.coord):
            if len(dim) != 1:
                mask[idx] = 1

        
        mask = list(map(bool,mask))
        
        # remove singleton dimension
        self.reduced_coord = list(compress(self.coord, mask))

        return mask

    def Crossection3D(self, interp_vals, phys_param=None):
        # Make Crossection3D
        
        print(len(self.new_dat))

        data_dim = len(np.shape(self.new_dat))

        # Check data has enough dimensions for 3D plotting
        if data_dim < 3:
            raise ValueError(f'Raw data only contains {data_dim}' + 
            ' non-singleton diemensions, ' + 
            'but 3 or more are required for Crossection3D method.')

        # Plot the values as a surface plot to depict
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        # Finda min/max values for the raw/interpolated data
        raw_bounds = [np.amin(self.new_dat), np.amax(self.new_dat)]
        interp_bounds = [np.amin(self.interp_arr), np.amax(self.interp_arr)]

        min_, max_ = min([raw_bounds[0], interp_bounds[0]]), max([raw_bounds[1], interp_bounds[1]])

        # Scatter plot of raw data
        ax.scatter(xs = self.M[0], ys = self.M[1], zs = self.M[2], c=self.new_dat,
            alpha=0.25,
            cmap='viridis_r',
            marker='D',
            vmin=min_,
            vmax=max_, label='Raw')

        # Scatter plot of interpolated data
        surf = ax.scatter(xs = self.Mi[0], ys = self.Mi[1], zs = self.Mi[2], c=self.interp_arr,
            alpha=0.5,
            cmap='viridis_r',
            marker='o',
            vmin=min_,
            vmax=max_, label='Interpolation')

        fig.colorbar(surf).set_label(phys_param,rotation=270)

        ax.set_xlabel('v1')
        ax.set_ylabel('v2')
        ax.set_zlabel('w1')
        ax.legend()
        plt.show()


    def Crossection2D(self, phys_param=None):

        # Finda min/max values for the raw/interpolated data
        raw_bounds = [np.amin(self.new_dat), np.amax(self.new_dat)]
        interp_bounds = [np.amin(self.interp_arr), np.amax(self.interp_arr)]

        min_, max_ = min([raw_bounds[0], interp_bounds[0]]), max([raw_bounds[1], interp_bounds[1]])

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(self.M[0], self.M[1], c=self.new_dat, cmap='viridis_r',
                    marker='D', alpha=0.8, vmin=min_, vmax=max_ ,label='Raw')

        surf = ax.scatter(self.Mi[0], self.Mi[1], c=self.interp_arr, cmap='viridis_r',
                    marker='o', alpha=0.8, vmin=min_, vmax=max_, label='Interpolation')
                
        fig.colorbar(surf).set_label(phys_param,rotation=270)

        ax.set_xlabel('v1')
        ax.set_ylabel('v2')
        ax.set_zlabel('w1')
        ax.legend()
        plt.show()

    def Crossection1D(self, phys_param=None):

        plt.figure()
        plt.plot(self.coord[0], self.response[:,0,0])
        plt.plot()
        plt.xlabel('v1')
        plt.ylabel('gl')
        plt.show()


    def plot(self):
        # Plot the values as a surface plot to depict
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        surf = ax.scatter(xs = self.M[0], ys = self.M[1], zs = self.M[2], c=data['response'][0], alpha=0.25)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        # Plot the result
        xx, yy = np.meshgrid(self.interp_vals[0], self.interp_vals[1])
        ax.scatter(xs = xx, ys = yy, c= self.interp_arr[4,:], s=20)
        ax.scatter(self.interp_vals[1][0] * np.ones(self.interp_vals[2].shape), self.interp_vals[2], c=self.interp_arr[0,:], s=20)
                
        ax.set_xlabel('v1')
        ax.set_ylabel('v2')
        ax.set_zlabel('w1')
        plt.show()

if __name__ == "__main__":

    data = {'voltages': [], 'response': {}}
    scale = 5

    # Set up grid and array of values
    t = np.arange(scale)

    data['response'][0] = t + t[:, np.newaxis] + t[:, np.newaxis][:, np.newaxis] + t[:, np.newaxis][:, np.newaxis][:, np.newaxis]+ t[:, np.newaxis][:, np.newaxis][:, np.newaxis][:, np.newaxis]
    
    data['voltages'].append(np.linspace(-0.05, 1.5, scale))
    # Adding sigleton
    data['voltages'].append(np.linspace(-5,-5,1))
    data['voltages'].append(np.linspace(0.2, 0.5, scale))
    # Adding sigleton
    data['voltages'].append(np.linspace(5,5,1))
    data['voltages'].append(np.linspace(0.1, 0.7, scale))

    interp_obj = InterpolateND(data['response'][0], *data['voltages'])
    

    # interpatate 1 ----------------------------------------------------   
    interp_vals = []
    
    # X
    interp_vals.append(np.linspace(0.35, 1, 5))
    # W1
    interp_vals.append(np.linspace(-5,-5, 1)) 
    # Y
    interp_vals.append(np.linspace(0.35, 0.45, 5))
    # W2
    interp_vals.append(np.linspace(5,5, 1)) 
    # Z
    interp_vals.append(np.linspace(0.3, 0.35, 5)) 

    interp_obj(*interp_vals)

    # interp_obj.Crossection1D(phys_param='gl')
    interp_obj.Crossection2D(phys_param='gl')
    interp_obj.Crossection3D(interp_vals, phys_param='gl')

    # interp_obj.plot()


    # interpatate 2 ----------------------------------------------------
 