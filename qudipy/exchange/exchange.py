"""
Functions for calculating the many-electron schrodinger equation.

@author: simba
"""

import numpy as np
import math
import qudipy as qd
from qudipy.utils import nchoosek
from itertools import combinations, permutations, product
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import gamma, beta, comb
from scipy import sparse
from tqdm import tqdm

from sys import getsizeof
from os import getcwd
import time


def optimize_HO_omega(gparams, nx, ny=None, ecc=1.0, omega_guess=1E15, 
                      n_se_orbs=2, opt_tol=1E-7, consts=qd.Constants("vacuum")):
    '''
    Find an optimal choice of omega used when building a basis of harmonic 
    orbitals centered at the origin which are used to approximate the single 
    electron orbitals of a potential. Minimization is done using the BFGS 
    algorithm. If the algorithm fails, this function will return None.

    Parameters
    ----------
    gparams : GridParameters object
        Contains the grid and potential information.
    nx : int
        Number of modes along the x direction to include in the harmonic 
        orbital basis set.
        
    Keyword Arguments
    -----------------
    ny : int, optional
        Number of modes along the y direction to include in the harmonic 
        orbital basis set. The default is nx.
    ecc : float, optional
        Specify the eccentricity of the harmonic orbitals defined as 
        ecc = omega_y/omega_x. The default is 1.0 (omega_y = omega_x).
    omega_guess : float, optional
        Initial guess for omega in the optimization. If the grid was infinitely
        large and dense, then choice of omega_guess is not important. For finite
        grid sizes, a poor choice of omega can cause the harmonic orbitals 
        themselves to be larger than than the grid itself. This causes
        obvious orthogonality issues. Therefore, it is heavily suggested that
        omega_guess is supplied with a decent guess. The default is 1E15 which
        should be smaller than most reasonable grid spacings in SI units.
    n_se_orbs : int, optional
        Number of single electron orbitals to compare to when checking how well
        the harmonic orbitals approximate the single electron orbitals for a 
        given choice of omega. The default is 2.
    opt_tol : float, optional
        Optimality tolerance for the BFGS algorithm. The default is 1E-7.
    consts : Constants object, optional
        Specify the material system constants. The default is "vacuum".

    Returns
    -------
    opt_omega : float
        The optimal omega found. If the optimization could not be completed, 
        then None is returned.
    opt_HOs : n-d array
        The basis of optimal harmonic orbitals. The first dimension of the 
        array corresponds to the index of the harmonic orbital.

    '''
    
    # Set default behavior of ny=nx
    if ny is None and gparams.grid_type == '2D':
        ny = nx
    
    # First thing to do is find the first few single electron orbitals from
    # the potential landscape.
    __, se_orbitals = qd.qutils.solvers.solve_schrodinger_eq(consts, gparams, 
                                                             n_sols=n_se_orbs)

    def find_HO_wf_difference(curr_log_w):
        
        # Undo omega log
        curr_w = 10**curr_log_w
        
        # Find current basis of HOs
        curr_HOs = build_HO_basis(gparams, curr_w, nx, ny, ecc=ecc,
                                  consts=consts)
        
        # Get current overlap matrix between HOs and SE orbitals
        S_matrix = qd.qutils.math.find_overlap_matrix(gparams, se_orbitals, 
                                                       curr_HOs)
        
        # S contains all <SE_i|HO_j> inner products. If we have chosen a good 
        # omega, then SUM(|<SE_i|HO_j>|^2) will be close to 1. Therefore, we
        # want to maximize this value.
        min_condition = np.abs(1 - np.diag(S_matrix.conj().T @ S_matrix))
        # Average the min condition with respect to the number of SE orbitals
        # so the ideal min condition = 1
        min_condition = np.sum(min_condition) / n_se_orbs
        
        return min_condition
        
    # Do a search over the log of omega to improve minimization robustness. In
    # particular when changing between different material systems and dot
    # geometries and sizes.
    opt_result = minimize(fun=find_HO_wf_difference,
                             x0=np.log10(omega_guess), method='BFGS',
                             options={'gtol': opt_tol})
            
    # If optimization was successful, return optimal omega and optimal basis
    if 'success' in opt_result.message:
        print(opt_result.message)
        opt_omega = 10**opt_result.x
        
        opt_HOs = build_HO_basis(gparams, opt_omega, nx, ny, ecc=ecc,
                                  consts=consts)
        
        return opt_omega[0], opt_HOs
    
    # If optimization failed, return None
    else:
        print(opt_result.message)
        
        return None, None

def build_HO_basis(gparams, omega, nx, ny=0, ecc=1.0,
                   consts=qd.Constants('vacuum')):
    '''
    Build the basis of 1D or 2D harmonic orbitals centered at the origin of the
    coordinate system.

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid and potential information.
    omega : float
        Harmonic frequency of the harmonic orbital basis along the x direction. 
        By default, the one along y direction has the same value; a different
        value can be specified by the ecc argument (see below). 
    nx : int
        Number of modes along the x direction to include in the basis set.
        
    Keyword Arguments
    -----------------
    ny : int, optional
        Number of modes along the y direction to include in the basis set. 
        Only applicable if gparams is for a '2D' system. The default is 0.
    ecc : float, optional
        Specify the eccentricity of the harmonic orbitals defined as 
        ecc = omega_y/omega_x. The default is 1.0 (omega_y = omega_x).
    consts : Constants object, optional
        Specify the material system constants when building the harmonic
        orbitals. The default assumes 'vacuum' as the material system.

    Returns
    -------
    HOs : array
        The constructed harmonic orbital basis where the first axis of the 
        array corresponds to a different harmonic orbital (HOs[n,:] for 1D and
        HOs[n,:,:] for 2D). If gparams describes a 2D grid, then the harmonic
        orbitals are ordered first by y, then by x.

    '''
    
    # Initialize the array for storing all the created harmonic orbitals and
    # get corresponding harmonic confinements along x and y
    omega_x = omega
    # Inverse characteristic length: used for shorter expressions when building 
    # harmonic orbitals
    alpha_x = np.sqrt(consts.me * omega_x / consts.hbar)
    if gparams.grid_type == '1D':
        HOs = np.zeros((nx, gparams.nx), dtype=complex)
    elif gparams.grid_type == '2D':
        omega_y = omega_x * ecc
         # Used for shorter expressions when building harmonic orbitals
        alpha_y = np.sqrt(consts.me * omega_y / consts.hbar)
        HOs = np.zeros((nx*ny, gparams.ny, gparams.nx), dtype=complex)
    

    # Construct all of the Hermite polynomials we will use to build up the
    # full set of HOs. We will store each nth Hermite polynomial to make this
    # more efficient while using the recursion formula to find higher order
    # polynomials.
    def _get_hermite_n(n, hermite_sub1, hermite_sub2, coords):
        '''
        Helper function for finding the n_th hermite polynomial IF the previous
        two nth polynomials are known.

        Parameters
        ----------
        n : int
            Specify which nth hermite polynomial currently being calculated.
        hermite_sub1 : array
            The H_{n-1} hermite polynomial (if applicable).
        hermite_sub2 : array
            The H_{n-2} hermite polynomial (if applicable).
        coords : array
            coordinates for the current hermite polynomial.

        Returns
        -------
        array
            The H_n hermite polynomial.

        '''
        
        # Base case 0
        if n == 0:
            return np.ones(coords.size)
        # Base case 1
        elif n == 1:
            return 2*coords
        # All other cases
        else:
            return 2*coords*hermite_sub1 - 2*(n-1)*hermite_sub2
    
    # Construct all the hermite polynomials which we will use to build up the
    # full set of HOs.   
    # x first
    x_hermites = np.zeros((nx, gparams.nx), dtype=complex)    
    for idx in range(nx):
        if idx == 0:
            x_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_x*gparams.x)
        elif idx == 1:
            x_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_x*gparams.x)
        else:
            x_hermites[idx,:] = _get_hermite_n(idx, x_hermites[idx-1,:],
                                              x_hermites[idx-2,:],
                                              alpha_x*gparams.x)
    # y now (if applicable)
    if gparams.grid_type == '2D':
        y_hermites = np.zeros((ny, gparams.ny), dtype=complex)  
        for idx in range(ny):
            if idx == 0:
                y_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_y*gparams.y)
            elif idx == 1:
                y_hermites[idx,:] = _get_hermite_n(idx, [], [], alpha_y*gparams.y)
            else:
                y_hermites[idx,:] = _get_hermite_n(idx, y_hermites[idx-1,:],
                                                  y_hermites[idx-2,:],
                                                  alpha_y*gparams.y)

    # Now that the Hermite polynomials are built, construct the 1D harmonic
    # orbitals
    # x first
    x_HOs = np.zeros((nx, gparams.nx), dtype=complex)
    for idx in range(nx):
        # Build harmonic orbital
        coeff = 1 / np.sqrt(2**idx * math.factorial(idx)) * (alpha_x ** 2 / 
                                                             math.pi) ** (1/4)
        x_HOs[idx,:] = coeff * np.exp(-alpha_x**2 *
                                         gparams.x**2 / 2) * x_hermites[idx,:]
        
    # y now (if applicable)
    if gparams.grid_type == '2D':
        y_HOs = np.zeros((ny, gparams.ny), dtype=complex)
        for idx in range(ny):
            # Build harmonic orbital
            coeff = 1 / np.sqrt(2**idx * math.factorial(idx)) * (alpha_y ** 2 / 
                                                             math.pi) ** (1/4)
            y_HOs[idx,:] = coeff * np.exp(-alpha_y**2 * gparams.y**2
                                                     / 2) * y_hermites[idx,:]

    # If building for a 2D grid, build the 2D harmonic orbital states
    if gparams.grid_type == '1D':
        HOs = x_HOs
    elif gparams.grid_type == '2D':
        idx_cnt = 0 # Used for saving harmonic orbitals to HOs array
        for x_idx in range(nx):
            # Get current x harmonic orbital and convert to meshgrid format
            curr_x_HO = x_HOs[x_idx,:]
            curr_x_HO, _ = np.meshgrid(curr_x_HO,np.ones(gparams.ny))
            for y_idx in range(ny):
                # Get current y harmonic orbital and convert to meshgrid format
                curr_y_HO = y_HOs[y_idx,:]
                _, curr_y_HO = np.meshgrid(np.ones(gparams.nx),curr_y_HO)
                
                # Make 2D harmonic orbital
                HOs[idx_cnt,:,:] = curr_x_HO * curr_y_HO                
                idx_cnt += 1
                
    return HOs
        

#TODO Also, this is a good example of a code that should be "generalized" to 
# handle a Hamiltonian class
def basis_transform(gparams, new_basis, 
                              consts=qd.Constants('vacuum'), unitary=True,
                              ortho_basis=False):
    '''
    This function takes an inputted Hamiltonian (gparams) and does a
    transformation into a new basis (new_basis).

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid and potential information.
    new_basis : n-d array
        A multi-dimensional array corresponding to the basis we will transform
        the Hamiltonian into. The first dimension should correspond to the 
        index of each basis state (i.e. basis vectors should be stored as rows).
        
    Keyword Arguments
    -----------------
    consts : Constants object, optional
        Specify the material system constants. The default is "vacuum".
    unitary : bool
        Specify if you want the unitary transformation that performs the basis
        transformation to be returned. Requires evaluation of the eigenvalue 
        problem. The default is True.
    ortho_basis : bool
        Specify if the new_basis is orthogonal or not. When False, we will 
        calculate the overlap matrix during the transformation.  When True, no
        overlap matrix is calculated. If you know ahead of time that your basis
        is orthogonal, specifying True can save some computational overhead by 
        not calculating the overlap matrix. The default is False.

    Returns
    -------
    ham_new : 2d array
        The Hamiltonian written in the new basis.
    U : 2d array
        The unitary transformation that yields U*H*U^-1 = H' where H is the
        original Hamiltonian and H' is the Hamiltonian in the new basis. This
        is only returned when unitary=True.
    eig_ens : 1d array
        The eigenergies of the Hamiltonian written in the new basis.

    '''
    
    # First thing is to build the Hamiltonian
    if gparams.grid_type == '1D':
        ham = qd.qutils.solvers.build_1DSE_hamiltonian(consts, gparams)
    elif gparams.grid_type == '2D':
        ham = qd.qutils.solvers.build_2DSE_hamiltonian(consts, gparams)
        
    # Intialize Hamiltonian for the basis transformation
    n_basis_states = new_basis.shape[0]
    ham_new = np.zeros((n_basis_states,n_basis_states), dtype=complex)
    
    # Now rewrite Hamiltonian in new basis by evaluating inner products <i|H|j>
    # First upper triangular elements
    for i in range(n_basis_states):
        # Ket state
        state_R = new_basis[i,:]
        # Convert to NO if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_MG_to_NO(state_R)
            
        # Evaluate H|j>
        state_R = ham @ state_R
        
        # Convert back to MG if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_NO_to_MG(state_R)
            
        for j in range(i+1,n_basis_states):
            # Bra state
            state_L = new_basis[j,:]

            # Evaluate <i|H|j>
            ham_new[i,j] = qd.qutils.math.inner_prod(gparams, state_L, state_R)

    # Now lower triangular elements
    ham_new += ham_new.conj().T
    
    # Now diagonal elements
    for i in range(n_basis_states):
        # Ket and bra states
        state_R = new_basis[i,:]
        state_L = state_R
        
        # Convert ket to NO if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_MG_to_NO(state_R)
            
        # Evaluate H|i>
        state_R = ham @ state_R
        
        # Convert back to MG if a 2D state
        if gparams.grid_type == '2D':
            state_R = gparams.convert_NO_to_MG(state_R)

        # Evaluate <i|H|i>
        ham_new[i,i] = qd.qutils.math.inner_prod(gparams, state_L, state_R)
            
    # Correct any numerical issues and force Hamiltonian to be Hermitian
    ham_new = (ham_new + ham_new.conj().T)/2;
    
    # Now calculate the unitary transformation to convert H -> H' (if desired)
    # U H U^-1 = H'
    if unitary is False:
        return ham_new
    else:
        # If basis is declared to be orthogonal, assume overlap matrix
        # is identity, otherwise calculate it.
        if ortho_basis is True:
            eig_ens, U = eigh(ham_new)
        else:
            S_matrix = qd.qutils.math.find_overlap_matrix(gparams, new_basis,
                                                           new_basis)
            eig_ens, U = eigh(ham_new, S_matrix)

    # Sort eigenenergies and unitary from smallest to largest (although it
    # already should be sorted by eigh output)
    eig_ens = eig_ens[eig_ens.argsort()]
    U = U[:,eig_ens.argsort()]
    
    return ham_new, U, eig_ens
        
        
def __calc_origin_cme(na:int, ma:int, nb:int, mb:int, ng:int,
                         mg:int, nd:int, md:int):
    '''
    Helper function that calculates a matrix element of Coulomb
    interaction in the harmonic orbital basis 
    in the effective Rydberg units. 
    See formula 7 in the exchange interaction paper for definition: 
    https://www.overleaf.com/project/5e879589e03b60000124dce7
    (defined in SI units there)

    Parameters
    ----------
    na : integer
        n_alpha parameter of the harmonic bra-function 
    ma : integer
        m_alpha parameter of the harmonic bra-function 
    nb : integer
        n_beta parameter of the harmonic bra-function 
    mb : integer
        m_beta parameter of the harmonic bra-function 
    ng : integer
        n_gamma parameter of the harmonic ket-function 
    mg : integer
        m_gamma parameter of the harmonic ket-function 
    nd : integer
        n_delta parameter of the harmonic ket-function 
    md : integer
        m_delta parameter of the harmonic ket-function 

    Returns
    -------
    CME: float 
        The Coulomb matrix element. 
    '''
    
    cme = 0
    
    # Initialize a and p
    a0 = na + nb + ng + nd
    b = ma + mb + mg + md
    pInd0 = a0 + b
        
    # If a and b are both even, then the CME will be non-zero.  Otherwise,
    # it will be zero and we can just skip doing all of these loops.  Note
    # that we don't need to consider the -2*p_i terms in a and b because
    # those are even.
    if a0 % 2 != 0 or b % 2 != 0:
        return(cme)
    
    for p1 in range(min(na,nd)+1):
        coef1 = math.factorial(p1) * nchoosek(na,p1) * \
            nchoosek(nd, p1)
        
        # Continue building a and p
        a1 = a0 - 2 * p1;
        pInd1 = pInd0 - 2 * p1
        
        for p2 in range(min(ma,md)+1):
            coef2 = coef1 * math.factorial(p2) * nchoosek(ma,p2) * \
                nchoosek(md,p2)
            
            # Continue building p
            pInd2 = pInd1 - 2 * p2
            
            for p3 in range(min(nb,ng) + 1):
                coef3 = coef2 * math.factorial(p3) * nchoosek(nb,p3) * \
                    nchoosek(ng,p3)
                
                # Finish building a and continue building and p
                a = a1 - 2 * p3
                pInd3 = pInd2 - 2 * p3
    
                for p4 in range(min(mb,mg)+1):                 
                    coef4 = coef3 * math.factorial(p4) * \
                            nchoosek(mb,p4) * nchoosek(mg,p4)
                    
                    # Finish building p
                    p = (pInd3 - 2 * p4) / 2
                    
                    # Skip this sum term if 2p is odd
                    if p % 2 != 0:
                        continue
                    
                    # Calculate the CME
                    cme = cme + (-1)**p * coef4 * gamma(p + 1/2) *\
                        beta(p - ((a - 1)/2), (a + 1)/2)
    
    # Take care of the scalar coefficients
    globPhase = (-1) ** (nb + mb + ng + mg)
    cme *= 2 /(math.pi * math.sqrt(2)) * globPhase
    cme = cme / math.sqrt(math.factorial(na) * math.factorial(ma) *\
        math.factorial(nb) * math.factorial(mb) * math.factorial(ng) *\
        math.factorial(mg) * math.factorial(nd) * math.factorial(md))
        
    return cme 

        
def calc_origin_cme_matrix(nx, ny, omega=1.0, consts=qd.Constants("vacuum"), 
                           rydberg=True, save_dir=None):
    '''
    Calculates the Coulomb Matrix Elements for a harmonic orbital basis. 
    CMEs are calculated assuming omega = 1 and then appropriately scaled.

    Parameters
    ----------
    nx : int
        Number of harmonic orbitals along the x direction
    ny : int
        Number of harmonic orbitals along the x direction

        
    Keyword Arguments
    -----------------
    omega : float, optional
        Width of the harmonic well that defines the harmonic basis.
        The default is 1.0.
    consts : Constants object, optional
        Specify the material system constants. The default is "vacuum".
    rydberg : bool, optional
        Specify whether to use the Rydberg units instead of SI units. 
        The default is True.
    save_dir : string, optional
        Path to save the CME array as a numpy native binary .npy.
        Useful for faster calculations. The naming 
        convention is as follows: "CME_{nx}x{xy}.npy". If None is specified, 
        array is not saved. 
        The default is None.

    Returns
    -------
    CMEs : 2d float array
        Array of Coulomb matrix elements, grouped in a 2d form from 8d  
        for convenient matrix multiplication. 
        Compound indices along the rows:
        (n_alpha, m_alpha, m_beta, n_beta), and along the colums:
        (n_gamma, m_gamma, m_delta, n_delta), where the last index changes
        the fastest


    '''
    
    n_HOs = nx * ny
    CMEs = np.zeros([n_HOs**2, n_HOs**2], dtype='float32')
    
    time.sleep(0.5) # Needed for tqdm to work properly if a print statement
                    # is executed right before running this function.
    for row_idx in tqdm(range(n_HOs**2)):
        # Parse the row index to extract alpha and beta and the subsequent
        # harmonic modes for x and y (n and m respectively)
        alpha = row_idx // n_HOs
        n_alpha = alpha // ny
        m_alpha = alpha % ny
        
        beta = row_idx % n_HOs
        n_beta = beta // ny
        m_beta = beta % ny
        
        for col_idx in range(n_HOs**2):
            # Parse the row index to extract gamma and delta and the subsequent
            # harmonic modes for x and y (n and m respectively)
            gamma = col_idx // n_HOs
            n_gamma = gamma // ny
            m_gamma = gamma % ny
            
            delta = col_idx % n_HOs
            n_delta = delta // ny
            m_delta = delta % ny
            
            # Check if CME is 0. This will avoid the __calc function returning
            # 0 and needing to be inserted into the CME array.
            a = n_alpha + n_beta + n_gamma + n_delta
            b = m_alpha + m_beta + m_gamma + m_delta
                
            if a % 2 != 0 or b % 2 != 0:
                continue
            
            # because of the internal symmetries of the expression, we need to 
            # calculate only some of the matrix elements. The rest of the values
            # coincide with the calculated values, and can be only addressed
            # in the array

            if (row_idx <= col_idx and n_delta <= m_delta and m_gamma <= m_delta
                    and n_gamma <= n_delta and m_beta <= m_gamma  
                    and n_beta <= n_gamma and m_alpha <= m_delta
                    and n_alpha <= n_delta):
                     
                CMEs[row_idx, col_idx] = __calc_origin_cme(n_alpha, m_alpha, 
                                                       n_beta, m_beta, 
                                                       n_gamma, m_gamma, 
                                                       n_delta, m_delta)
            else:
                # calculating the lower triangular part from the upper 
                # triangular one in a memory-efficient way
                if row_idx > col_idx:
                    n_alpha, m_alpha, n_beta, m_beta,  \
                        n_gamma, m_gamma, n_delta, m_delta =\
                            n_gamma, m_gamma, n_delta, m_delta, \
                                n_alpha, m_alpha, n_beta, m_beta
                        
                if n_delta > m_delta:       #change all m_i <-> n_i
                    n_alpha, m_alpha, n_beta, m_beta,  \
                        n_gamma, m_gamma, n_delta, m_delta = \
                            m_alpha, n_alpha, m_beta, n_beta, \
                                m_gamma, n_gamma, m_delta, n_delta

                if m_gamma > m_delta:
                    m_alpha, m_beta, m_gamma, m_delta = \
                        m_beta, m_alpha, m_delta, m_gamma

                if n_gamma > n_delta: 
                    n_alpha, n_beta, n_gamma, n_delta = \
                        n_beta, n_alpha, n_delta, n_gamma

                if m_beta > m_gamma:
                    m_beta, m_gamma = m_gamma, m_beta

                if n_beta > n_gamma:
                    n_beta, n_gamma = n_gamma, n_beta

                if m_alpha > m_delta:
                    m_alpha, m_delta = m_delta, m_alpha

                if n_alpha > n_delta:
                    n_alpha, n_delta = n_delta, n_alpha

                #indices of the symmetrical value in the matrix
                symm_row_idx = m_beta + n_beta * ny \
                                    + n_HOs * ( m_alpha + n_alpha * ny) 

                symm_col_idx = m_delta + n_delta * ny \
                                    + n_HOs * ( m_gamma + n_gamma * ny)

                CMEs[row_idx, col_idx] = CMEs[symm_row_idx, symm_col_idx]


    # We save only the matrix in Rydberg units for omega=1.0 by default.
    # This makes the matrix universal for all material systems. 
    # The appropriate scaling will be done elsewhere if the library is loaded

    if save_dir is not None: 
        with open(save_dir+f'\\CMEs_{nx}x{ny}.npy', 'wb') as f:
            np.save(f, CMEs)

    # If SI units, we need to scale CMEs by k 
    if not rydberg:
        CMEs *= (consts.e**2 / (8 * consts.pi * consts.eps) *
              np.sqrt(consts.me / consts.hbar ))
                        
    # Scale by omega if not the default value
    CMEs = CMEs if omega == 1.0 else CMEs * math.sqrt(omega)
    

    return CMEs
    
        
        
def build_fock_basis(n_elec: int, n_se_orbs: int, spin_subspace='all'):   
    '''
    Build the many electron spin-orbit basis vectors in Fock space.
    
    Parameters:
        n_elec: int
            Number of electrons in the system.
        n_se_orbs: int
            Number of single electron orbitals to use when constructing the spin
            orbit subspace.

    Keyword arguments:
        spin_subspace: int 1d array/iterable, or string
            Specifies which subspaces of the total spin projection operator S_z
            to use when constructing the 2nd quantization Hamiltonian. Limiting 
            the spin subspace significantly speeds up the calculation of the 
            energy eigenvalues, since the Heisenberg Hamiltonian commutes with 
            S_z, which leads to degeneracy. Usually, [0] for even n_elec or 
            [0.5] for odd n_elec is sufficient to obtain 
            all distinct energy values. 
            
            As an example, for a 3 electron system, 
            there are four possible S_z values [-1.5, -0.5, 0.5, 1.5]. To use  
            only S_z > 0, set spin_subspace = [1/2, 3/2].
            To use all spin subspaces, set spin_subspace='all'.
            The default is 'all'.

    Returns
    -------
    fock_so_basis, fock_so_basis_bool : int 1D array, and sparse bool 2D array
        Compilation of all many-electron spin-orbit basis states in Fock space
        within the specified spin subspace. 

        Each of the rows of the fock_so_basis_bool matrix
        is a boolean 1D array of length 2 * n_se_orbs
        with the total number of logical '1's equal to n_elec.
        First half of the row corresponds to spin-down states, whereas the second 
        half is vice versa. 
        e.g. array([1,0,0,1,0,1,0,0], dtype=bool) for n_se_orbs=4, n_elec=3:
        1 electron in ground state, spin down
        1 electron in the 3rd excited state, spin down
        1 electron in the 1st excited state, spin up
        Each row is packed into an integer number as follows: 
        [0,0,1,1] -> 3, [0,1,0,1] -> 5, and stored in the 1D integer array 
        fock_so_basis. 

        Both the integer and boolean representations are returned. Mind that 
        fock_so_basis_bool can be reconstructed from fock_so_basis only if 
        n_se_orbs is known a priori. 

    '''       
        # building the spin subspace 
    
    timer = time.time()

    if isinstance(spin_subspace, str) and spin_subspace.lower() == 'all':
        spin_subspace = np.arange(- n_elec / 2, n_elec / 2 + 1, 1 )
        
    else:
        # set removes repeated values
        spin_subspace = np.array(list(set(spin_subspace)))

    # checking if the number format of spin_subspace is correct, i.e
    # there are only integers for even n_elec or only half integers for odd n_elec

    denom = 2 if n_elec % 2 else 1  # gives 2 if S_z values are 1/2, 3/2, etc.

    ints_or_half_ints = (np.array_equal(spin_subspace * denom, 
                                    (spin_subspace * denom).astype(int)) &
                            np.all((spin_subspace * 2 + denom) % 2)
                            )
    if not ints_or_half_ints:
        raise ValueError('Incorrect format of the input, please try again. '+
        'All numbers must be either integers (for even n_elec)' 
        +' or half integers (for odd n_elec) simultaneously')
    
    # checking if the spin subspace can be realized in such systems
    if np.any(np.greater(np.abs(spin_subspace), n_elec / 2)):
        raise ValueError(f'The system with {n_elec} electrons cannot have '+
                            'the specified total spin(s). Please try again')


    # creating empty arrays where Fock states will be collected in the 
    # uncompressed and compressed format (with the Fock vector being a boolean 
    # array and the binary representation of an integer, respectively)

    fock_so_basis_bool = np.empty((0, 2 * n_se_orbs), dtype=bool)

    fock_so_basis = np.empty((0,), dtype=int)
    # calculating the number of spins up and spins down for each spin 
    # subspace

    for sz in spin_subspace:
        # For each S_z value, the numbers of electrons in spin-up and spin-down 
        # states are obtained from N(↑) - N(↓) = 2 S_z, N(↑) + N(↓) = n_elec

        n_down = int(n_elec / 2 - sz)
        n_up = int(sz + n_elec / 2)

        # numbers of multi-electron spin-orbit states 
        n_combs_down = comb(n_se_orbs, n_down, exact=True)
        n_combs_up = comb(n_se_orbs, n_up, exact=True)

        n_combs = n_combs_up * n_combs_down

        # a Fock state subarray for this subspace
        fock_subset_bool = np.zeros((n_combs, 2 * n_se_orbs), dtype=bool)

        # creating the arrays with column indexes of logical '1's  
        # for spins up and down separately.

        ones_ids_up = (np.array(list(combinations(range(n_se_orbs), 
                                            n_up)), dtype=int) )

        ones_ids_down = np.array(list(combinations(range(n_se_orbs), 
                                            n_down)), dtype=int) + n_se_orbs
        # the shift by n_se_orbs is necessary because all spin-up
        # information is stored in the left half of Fock vector,
        # and we read integers represented as bitsets right-to-left, 
        # not left-to-right
    
        # repeating each of the arrays an appropriate number of times 
        # to concatenate them and obtain all possible combinations of 
        # electrons with spins up and down
        # (similar to itertools.product of the arrays)

        ones_ids_down = np.broadcast_to(ones_ids_down, 
                            (n_combs_up, )+ ones_ids_down.shape).swapaxes(0,1)
       
        ones_ids_up = np.broadcast_to(ones_ids_up, 
                            (n_combs_down, ) + ones_ids_up.shape)

        ###ones_ids_down = np.repeat(ones_ids_down, n_combs_up, axis=0)
        ###ones_ids_up = np.tile(ones_ids_up, (n_combs_down,1))

        #concatenate spin-up and spin-down electron positions, and flatten the 
        # first two dimensions

        ones_ids = np.concatenate((ones_ids_down, ones_ids_up), 
                                axis=2).reshape(n_combs, n_elec)

        ###ones_ids = np.concatenate((ones_ids_down, ones_ids_up), axis=1)

        # creating the Fock space vector subset corresponding to the 
        # specified S_z value:
        # interpreting each row as a bitset and converting each of them
        # into an integer

        fock_subset = np.sum(np.left_shift(1, ones_ids), axis=1, 
                                    dtype=int)

        
        fock_subset_bool[np.arange(n_combs).reshape(n_combs,1), ones_ids] = 1

        #print('fock subset /ones ids\n', fock_subset # fock_subset)

        # appending the array of all Fock states
        fock_so_basis = np.append(fock_so_basis, fock_subset, axis=0)
        fock_so_basis_bool = np.append(fock_so_basis_bool, 
                                            fock_subset_bool, axis=0)


    print(f'Fock basis of length {fock_so_basis.shape[0]} created, '
                                f'time elapsed:{time.time()-timer} s' )

    return fock_so_basis, sparse.csr_matrix(fock_so_basis_bool)

def get_fock_spin_config(fock_so_basis_bool):
    """
    Get the spin configuration of a Fock spin-orbit basis in boolean 
    representation.
    
    Parameters:
        fock_so_basis_bool: 2D sparse matrix
            Fock spin-orbit basis in boolean representation, 
            with the rows of type |0 0 1 0 1 0 0 1>

    Returns
    -------
    fock_spin_config: 1D array
        Spin configuration: total number of spin-up states in each Fock vector

    """
    # inferring the parameters from the supplied basis
    basis_size, n_so = fock_so_basis_bool.shape
    n_se_orbs = n_so // 2
    # finding the number of electrons in each configuration and verifying they 
    # are the same
    nums_elec = fock_so_basis_bool.sum(axis=1, dtype=int)
    n_elec = nums_elec[0,0]
    if np.any(nums_elec - n_elec):
        raise ValueError('The Fock basis is constructed incorrectly. '
                            'Verify that the total number of '
                            'electrons is the same in each state')


    # halving the rows, summing up each half, reshaping back
    spins_down_up = np.asarray(fock_so_basis_bool.reshape((2 * basis_size,
                                            n_se_orbs)).sum(axis=1, dtype=int) 
                                                ).reshape(basis_size,2)
    fock_spin_config = spins_down_up[:,1]  # total numbers of spin up states
    return fock_spin_config

    

            
def annihilation_mask(fock_so_basis, fock_so_basis_bool):
    """
    Build the mask which shows what operators of the form 
    c_j^\dag c_i^\dag c_k c_l, when applied, give nonzero elements.
    
    Parameters:
        fock_so_basis: 2D array
            Fock basis in bitset representation
        fock_so_basis_bool: 2D sparse matrix
            Fock basis in boolean representation
    Returns
    -------
    annihilation_mask: 2D boolean array
        Mask which shows which CMEs are to be used to construct the
        second quantization Hamiltonian
    """

    # total size of the basis, and number of single electron 
    # spin-orbit wavefunctions
    basis_size, n_so = fock_so_basis_bool.shape
    n_se_orbs = n_so // 2

    kl_s = np.arange(n_so **2)
    k_s, l_s = (kl_s // n_so, kl_s % n_so)      # l index changes fastest

    # creating a bit mask annihilating an individual 
    # spin-orbit basis vector |k, s>, |l, s>

    k_bit_mask = np.left_shift(1, k_s)
    l_bit_mask = np.left_shift(1, l_s)
    
    kl_bit_mask = l_bit_mask  ^ k_bit_mask  # xor ensures that the application 
                                            # of c_k c_l gives zero when k==l                                           # 

    timer = time.time()

    # equivalence check is to ensure that both electrons were annihilated
    # multiplication is to ensure that the bit mask is nonzero
    # kl_mask = (((fock_so_basis[:, np.newaxis] & kl_bit_mask ) == kl_bit_mask)
    #                     & np.greater(kl_bit_mask,0) )

    kl_mask = sparse.coo_matrix(
       ( (fock_so_basis[:, np.newaxis] & kl_bit_mask ) == kl_bit_mask)
                        * kl_bit_mask, dtype=bool)

    # the ji mask is analogous, so the total mask before comparison of bra
    # and ket vectors:
    # annihilation_mask = np.kron(kl_mask, kl_mask)
    applicability_mask = sparse.kron(kl_mask, kl_mask, format='bsr')

    print(f'Applicability mask created within {time.time()-timer} s')  

    kl_mask_coords = (kl_mask.row, kl_mask.col)

    ########## Electron scattering on Coulomb potential ###############

    # constructing the "annihilated" bitsets by flipping bits where the 
    # mask of c_k c_l applicability indicates

    timer = time.time() 
    kl_bitsets = sparse.coo_matrix(
                        ((fock_so_basis[:, np.newaxis] 
                            ^ kl_bit_mask)[kl_mask_coords], kl_mask_coords), 
                                        shape=(basis_size, n_so**2)).tocsr()
                    
    print(f'Active bitsets grouped within {time.time()-timer} s')  
    # masks on the left and on the right are very similar in structure. 
    # size of each is (basis_size)**2 x (n_so)**4
    # 
    # ons = np.ones((basis_size, n_so**2), dtype=int)
    timer = time.time() 

    right_bitsets = sparse.kron(kl_mask, kl_bitsets, 
                                    format='bsr') # elements change faster
    left_bitsets = sparse.kron(kl_bitsets, kl_mask, 
                                    format='bsr') # elements change more slowly

    print(f'Orbital Kronecker products built within {time.time()-timer} s') 

    # bra and ket comparison: 
    # elements to exclude from the total annihilation mask because of 
    # the incompatibility of spin-orbit states for scattering
    timer = time.time() 
    orb_excl_mask = right_bitsets != left_bitsets
        
    print(f'Orbital exclusion  matrix built within {time.time()-timer} s')

    ############## Checking spin conservation upon scattering ############## 

    # spin states extracted from spin-orbit states
    x_k, x_l =  k_s // n_se_orbs, l_s // n_se_orbs
    x_kl = (2 * x_k + x_l).astype(np.uint8)

    # the spin mask is the same for all many-electron states, i.e. all rows
    timer = time.time()
    kl_spins = sparse.coo_matrix((x_kl[kl_mask.col], kl_mask_coords), 
                                        shape=(basis_size, n_so**2)).tocsr()
    print(f'Active spin configurations grouped within {time.time()-timer} s') 

    timer = time.time()
    right_spins = sparse.kron(kl_mask, kl_spins, 
                                    format='bsr') # elements change faster
    left_spins = sparse.kron(kl_spins, kl_mask, 
                                    format='bsr') # elements change more slowly
    print(f'Spin Kronecker products built within {time.time()-timer} s') 

    # states to exclude from the annihilation matrix because of
    # spin incompatibility
    spin_excl_mask = right_spins != left_spins

    # zeroing the values in the annihilation mask that are included in the 
    # exclusion masks   

    timer = time.time() 
    annihilation_mask = (applicability_mask > spin_excl_mask) > orb_excl_mask
    print(f'Annihilation mask constructed within {time.time()-timer} s')

    # conversion for fast summation
    return annihilation_mask.tobsr()       #### .tocsr()

def build_sq_ham(n_elec: int, spin_subspace, n_se_orbs: int, 
                           se_energies, se_cmes):
    '''
    Builds the second quantization Hamiltonian.  

    Parameters:
        n_elec: int
            Number of electrons in the system.
        spin_subspace: int 1d array/iterable, or string
        Specifies which sping subspaces to use when constructing the 2nd 
        quantization Hamiltonian. As an example, for a 3 electron system, there
        are four possible S_z values [-1.5, -0.5, 0.5, 1.5]. To use only 
        S_z > 0, set spin_subspace = [2,3] corresponding to the 3rd and 4th 
        elements of the array. To use all spin subspaces, set spin_subspace='all'.
        The default is 'all'.
        n_se_orbs: int
            Number of single-electron orbitals basis states.
        se_energies: float 1d  array
            Corresponding single electron orbital basis state eigenenergies.
        se_cmes : double 2D array
            Coulomb matrix elements in the single electron basis.

    Returns
    -------
    ham_sq : double 2D array
        Second quantization Hamiltonian.

    '''
    #constructing the Fock basis
    fock, fock_bool = build_fock_basis(n_elec, n_se_orbs, spin_subspace)
    n_so = n_se_orbs *2
    basis_size = fock.shape[0]
    
    #spin-up and spin-down single-electron energies are the same

    se_energies_spin = np.tile(np.flip(se_energies), 2)

    ens_matrix = fock_bool.multiply(se_energies_spin)

    # total energies of each Fock state ignoring interactions 
    # ravelled to a 1D array

    ens_noninter = np.asarray(ens_matrix.sum(axis=1)).ravel()
    
    # building up the diagonal part of the Hamiltonian

    ham_0 = np.diag(ens_noninter)

    ########## building up the nondiagonal part ############

    # introducing the summation indices extracted from the compound
    # index (orbital + spin DoF)

    ijkl_s = np.arange(n_so ** 4, dtype=int)
    ij_s, kl_s = ijkl_s // n_so ** 2, ijkl_s % n_so ** 2
    
    i_s, j_s = ij_s // n_so, ij_s % n_so # i changes slowest, j is 2nd slowest
    k_s, l_s = kl_s // n_so, kl_s % n_so # k is 2nd fastest, l is the fastest

    # orbital states without spins    
    i, j, k, l = i_s % 2, j_s % 2, k_s % 2, l_s % 2

    ij = n_se_orbs * i + j    
    kl = n_se_orbs * k + l       
    ijkl = n_se_orbs**2 * ij + kl

    # array of phases for a flattened CME matrix
    phase = (-1) ** (i + j + k + l)
    
    # creating a Coulomb interaction part of the Hamiltonian
    timer = time.time()

    ann_mask = annihilation_mask(fock, fock_bool)
    
    print(f'Annihilation mask of size {ann_mask.shape}',   
    f' in {ann_mask.format} format has been created ',
        f'successfully within {time.time()-timer} sec')
    # CMEs that give nonzero contribution to the Hamiltonian
    # scaled by an appropriate phase
    timer = time.time()
    phased_cmes = ann_mask.multiply(se_cmes.ravel()[ijkl]).multiply(phase* 0.5) # 

    print(f'Matrix of scaled CMEs of shape {phased_cmes.shape} in',
     f'{phased_cmes.format} format has been created ',
        f'successfully within {time.time()-timer} sec')
    # the nondiagonal part of the Hamiltonian due to Coulomb interaction
    ham_coul = phased_cmes.sum(axis=1).reshape(basis_size, basis_size)
    
    ham_sq = ham_0 + ham_coul
    return ham_sq


    
    
#   # mask for c_l c_k
#     right_mask = np.empty((basis_size, ) + 2 * (n_so, ), dtype=bool)

#     # building the grid of indices

#     c_l, c_k = np.ogrid[:n_so, :n_so]

#     # the annihilation operators give True value only when the corresponding 
#     # single-electron state is True within the Fock state
#     # moreover, when l=k, the result must be False

#     right_mask[:, c_l, c_k] = (fock_so_basis[:, c_l] & fock_so_basis[:, c_k]
#                                 & np.invert(np.eye(n_so,dtype=bool))
#                                 )
    
#     right_mask = sparse.csr_matrix( 
#                                 right_mask.reshape((basis_size, (n_so) ** 2 )))

#     # the conjugate "left mask" for (a_j a_i)^\dag is the same
#     applicabiblity_mask = sparse.kron(right_mask, right_mask)



if __name__ == "__main__":
    n_elec=3
    n_se_orbs=10
    spin_subspace=[0.5]

    ff, ff_bool = build_fock_basis(n_elec, n_se_orbs, spin_subspace)
    #print('Fock space:\n', ff)
    print('Fock space byte size:\n', getsizeof(ff), 
                '\nNumber of basis vectors:\n', ff.shape[0])        
    
    # masks = (single_annihilation_mask(braket_idx, ff, n_se_orbs) 
    #            for braket_idx in tqdm(range((ff.shape[0])**2)) )

    # timer = time.time()
    # print('Beginning stacking the matrices')
    # stacked_mm = sparse.vstack(masks,format='bsr', dtype=bool)
    # print('Time to stack masks:', time.time()-timer) 
    # # print('stacked mask',   stacked_mm)
    # print('stacked mask data size', stacked_mm.data.shape[0])

    mm = annihilation_mask(ff, ff_bool)
    print("mask data size: ", mm.data.nbytes, '\n mask\n', mm,) #mm.nonzero()
   
    # print('Unequal elements:',  (mm != stacked_mm).nnz )
    
    

    



            






    
        
        
        