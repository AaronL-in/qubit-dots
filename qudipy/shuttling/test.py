import os, sys
sys.path.append('../../')

import qudipy as qd
import qudipy.potential as pot
import qudipy.qutils as qt
import numpy as np
from numpy.fft import fft, ifft, fftshift, ifftshift
from scipy import sparse
from scipy.sparse import diags
from scipy.linalg import expm

import matplotlib.pyplot as plt

def initialize_params():
    """
    initialize the constants class with the Si/SiO2 material system and the GridParameters obejct 
    of the harmonic potential
    """
    # Initialize the constants class with the Si/SiO2 material system 
    consts = qd.Constants("Si/SiO2")

    # First define the x-coordinates
    x = np.linspace(-100,100,256)*1E-9
    # Define harmonic oscillator frequency
    omega = 5E12
    sep = 27.25E-9
    # Now construct the harmonic potential
    harm_pot = 1/2*consts.me*omega**2*np.square(x)
    
    # Create a GridParameters object
    gparams = pot.GridParameters(x, potential=harm_pot)

    return consts, gparams

def initialize_wf(consts, gparams):
    """
    find the initial wavefunction psi, which is a 1D array of dimension nx_local
    using the constants class with the Si/SiO2 material system and the GridParameters obejct 
    of the harmonic potential
    TODO: generalize to 2D wavefunctions
    """
    # Pass sparams, gparams to the solve_schrodinger_eq qutils method to obtain the eigenvalues and eigenvectors
    e_ens, e_vecs = qt.solvers.solve_schrodinger_eq(consts, gparams, n_sols=5)      # n_sols set to 0 to obtain ground state
    # psi = np.real(e_vecs[:,0])
    print("energy 0: ", e_ens[0])
    print("energy 1: ", e_ens[1])
    print("energy dff:", e_ens[1] - e_ens[0])
    t_time = 1/(2*(e_ens[1] - e_ens[0])/6.626E-34)
    print("tunnel time [s]:", t_time)

    psi = e_vecs[:,0]
    print('Norm psi: ', qd.qutils.math.inner_prod(gparams, psi, psi))

    return psi, t_time

def main():
    # initialize relevant constants and parameters for the calculation
    consts, gparams = initialize_params()
    # diagonal matrix of potential energy in position space
    PE_1D = gparams.potential
    
    dt = 5E-16
    # vector of position grid
    X = gparams.x                

    # indices of grid points
    I = [(idx-gparams.nx/2) for idx in range(gparams.nx)]   
    # vector of momentum grid
    P = np.asarray([2 * consts.pi * consts.hbar * i / (gparams.nx*gparams.dx) for i in I])

    # diagonal matrix of kinetic energy in momentum space
    #KE_1D = np.multiply(P,P)/(2*consts.me)

    # exponents present in evolution
    exp_K = np.exp(-1j*dt/2*np.multiply(P,P)/(2*consts.me*consts.hbar))
    exp_KK = np.multiply(exp_K,exp_K)
    exp_P = np.exp(-1j*dt/consts.hbar*gparams.potential)

    # initialize psi(t=0)
    psi_x, t_time = initialize_wf(consts, gparams)
    # print("initial: ", psi_x)
    # print("initial probability is: ", [abs(x)**2 for x in psi_x])
    print("Plotting the initial wavefunction...")
    plt.plot(X, [abs(x)**2 for x in psi_x])
    # plt.plot(X, psi_x)
    plt.show()

    # print(consts.me)
    # print(consts.hbar)
    # print(P)
    # print(exp_K)
    
    psi_p = fftshift(fft(psi_x))
    psi_p = np.multiply(exp_K,psi_p)
    
    # iterate through nprint time steps
    # number of time steps
    print(t_time)
    nt = int(np.round(t_time/dt))
    print("Number of time steps:",nt)
    # nt = 20000
    for step in range(nt):
        psi_x = ifft(ifftshift(psi_p))     
        psi_x = np.multiply(exp_P,psi_x)
        
        psi_p = fftshift(fft(psi_x))     
        
        if step != nt-1:
            psi_p = np.multiply(exp_KK,psi_p)
        else:
            psi_p = np.multiply(exp_K,psi_p)
            psi_x = ifft(ifftshift(psi_p))

    output = psi_x
    print("output norm:", qd.qutils.math.inner_prod(gparams,psi_x,psi_x))
    #print("output: ", output)
    # print("the resultant probability is: ", [abs(x)**2 for x in output])
    print("Plotting the wavefunction at time ",nt * dt)
    plt.plot(X, [abs(x)**2 for x in output])
    plt.show() 

    


main()