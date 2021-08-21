"""
Functions for calculating the many-electron schrodinger equation.

@author: simba
"""

import numpy as np
import math
import qudipy as qd
import itertools
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import gamma, beta
from tqdm import tqdm
import time
import os

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
        

# Also, this is a good example of a code that should be "generalized" to 
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
                         mg:int, nd:int, md:int, rydberg=False):
    '''
    Helper function that calculates the matrix elements of Coulomb
    interaction in the harmonic orbital basis. 
    See formula 7 in the exchange interaction paper for definition: 
    https://www.overleaf.com/project/5e879589e03b60000124dce7
    The function assumes SI units by default. 

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
        
    Keyword Arguments
    -----------------
    rydberg : bool
        Specifies whether the result should be outputted in the effective
        Rydberg units rather than SI. False by default.

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
        coef1 = math.factorial(p1) * qd.utils.nchoosek(na,p1) * \
            qd.utils.nchoosek(nd, p1)
        
        # Continue building a and p
        a1 = a0 - 2 * p1;
        pInd1 = pInd0 - 2 * p1
        
        for p2 in range(min(ma,md)+1):
            coef2 = coef1 * math.factorial(p2) * qd.utils.nchoosek(ma,p2) * \
                qd.utils.nchoosek(md,p2)
            
            # Continue building p
            pInd2 = pInd1 - 2 * p2
            
            for p3 in range(min(nb,ng) + 1):
                coef3 = coef2 * math.factorial(p3) * qd.utils.nchoosek(nb,p3) * \
                    qd.utils.nchoosek(ng,p3)
                
                # Finish building a and continue building and p
                a = a1 - 2 * p3
                pInd3 = pInd2 - 2 * p3
    
                for p4 in range(min(mb,mg)+1):                 
                    coef4 = coef3 * math.factorial(p4) * \
                            qd.utils.nchoosek(mb,p4) * qd.utils.nchoosek(mg,p4)
                    
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
    cme = cme/(math.pi * math.sqrt(2)) * globPhase
    cme = cme / math.sqrt(math.factorial(na) * math.factorial(ma) *\
        math.factorial(nb) * math.factorial(mb) * math.factorial(ng) *\
        math.factorial(mg) * math.factorial(nd) * math.factorial(md))
        
    # If effective Rydberg units, then multiply by 2
    if rydberg:
        cme = 2 * cme
        
    return cme 

        
def calc_origin_cme_matrix(nx, ny, omega=1.0, consts=qd.Constants("vacuum"), 
                           rydberg=False, save_dir=os.getcwd()):
    '''
    Calculates the Coulomb Matrix Elements for a harmonic orbital basis. CMEs
    are calculated assuming omega = 1 and then appropriately scaled.

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
        Specify whether to replace the default SI units with the Rydberg units. 
        The default is False.
    save_dir : string, optional
        Path to save the CME array as a numpy native binary .npy.
        Useful for faster calculations. The naming 
        convention is as follows: "CME_{nx}x{xy}.npy". If None is specified, 
        array is not saved. 
        The default is the current working directory.

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
    CMEs = np.zeros([n_HOs**2, n_HOs**2])
    
    time.sleep(0.5) # Needed for tqdm to work properly if a print statement
                    # is executed right before running this function.
    for row_idx in tqdm(range(n_HOs**2)):
        # Parse the row index to extract alpha and beta and the subsequent
        # harmonic modes for x and y (n and m respectively)
        alpha = math.floor(row_idx / n_HOs)
        n_alpha = math.floor(alpha / ny)
        m_alpha = alpha % ny
        
        beta = row_idx % n_HOs
        n_beta = math.floor(beta / ny)
        m_beta = beta % ny
        
        for col_idx in range(row_idx,n_HOs**2):
            # Parse the row index to extract gamma and delta and the subsequent
            # harmonic modes for x and y (n and m respectively)
            gamma = math.floor(col_idx / n_HOs)
            n_gamma = math.floor(gamma / ny)
            m_gamma = gamma % ny
            
            delta = col_idx % n_HOs
            n_delta = math.floor(delta / ny)
            m_delta = delta % ny
            
            # Check if CME is 0. This will avoid the __calc function returning
            # 0 and needing to be inserted into the CME array.
            a = n_alpha + n_beta + n_gamma + n_delta
            b = m_alpha + m_beta + m_gamma + m_delta
                
            if a % 2 != 0 or b % 2 != 0:
                continue
            
            # Get the corresponding CME
            CMEs[row_idx, col_idx] = __calc_origin_cme(n_alpha, m_alpha, 
                                                       n_beta, m_beta, 
                                                       n_gamma, m_gamma, 
                                                       n_delta, m_delta,
                                                       rydberg)
            
    # We only found the upper triangular part of the matrix so find the
    # lower triangular part here
    temp = CMEs - np.diag(np.diag(CMEs))
    CMEs = CMEs + temp.T # CME is real so no need for .conj()
        
    # Now scale CME matrix if appropriate
    # If effective Rydberg units, then no need to scale CME   
    if rydberg:
        k = 1
    # Otherwise we have SI units and need to scale CMEs by k
    else:
        k = consts.e**2 / (4 * consts.pi * consts.eps) *\
            np.sqrt(consts.me / consts.hbar)
        
    # Scale by k
    CMEs *= k
    # Scale by omega if not the default value
    CMEs = CMEs if omega == 1.0 else CMEs*math.sqrt(omega)
    
    if save_dir is not None: 
        with open(save_dir+f'\\CMEs_{nx}x{ny}.npy', 'wb') as f:
            np.save(f, CMEs)
    return CMEs
    

def build_so_basis_vectors(n_elec: int, spin_subspace, n_se_orbs: int):
    '''
    Build the many electron spin orbit basis vectors. These can be used to map
    the output of the many electron eigenvectors to the corresponding many
    electron spatial wavefunctions and spin states.

    Parameters
    ----------
    n_elec : int
        Number of electrons.
    spin_subspace : int array
        Specifies which sping subspaces to use when constructing the 2nd 
        quantization Hamiltonian. As an example, for a 3 electron system, there
        are four possible S_z values [-1.5, -0.5, 0.5, 1.5]. To use only 
        S_z > 0, set spin_subspace=[2,3] corresponding to the 3rd and 4th 
        elemtns of the array. To use all spin subspaces, set spin_subspace='all'.
        The default is 'all'.
    n_se_orbs : int
        Number of single electron orbitals to use when constructing the spin
        orbit subspace.

    Returns
    -------
    vec_so_basis : int 2D array
        Compilation of all many electron spin orbit basis states. First index
        corresponds to the ith state and the other indices are for the individual
        electron states for that given many electron state. Format is as follows:
        The first K = n_elec indicies correspond to the orbital state and the last
        K = n_elec indicies correspond to the spin state.
        As an explicit example for a K = 3 case, consider the multi-electron 
        spin-orbit state [4,2,3,0,0,1] which means:
        1st electron is in the 4th orbital state (idx=0) with spin down (idx=3)
        2nd electron is in the 2nd orbital state (idx=1) with spin down (idx=4)
        3rd electron is in the 3rd orbital state (idx=2) with spin up   (idx=5)
    map_so_basis : int 2D array
        A 2D array which maps the ith single electron spin orbit state (first
        index) to the corresponding single electron orbital and spin state.

    '''
    # Parse input, and convert to numpy array
    if spin_subspace == 'all':
        spin_subspace = np.array(range(n_elec+1))
    else:
        spin_subspace = np.array(spin_subspace)
    
    # Check to see if spin subspace array is valid or not (S_z indices
    # cannot exceed n_elec or be less than 0.
    if min(spin_subspace) < 0 or max(spin_subspace) > n_elec:
        raise ValueError("Spin subspace indices must be positive integers and"+
                         " less than n_elec+1.")
    # Otherwise check that all supplied indices are integers
    else:
        for idx in spin_subspace:
            if np.floor(idx) != idx:
                raise TypeError("Spin subspace indices must be positive integers"+
                                " and less than n_elec+6.")
         
    # Get total number of single electron spin-orbit states
    n_se_so = 2 * n_se_orbs
    if n_elec > n_se_so:
        raise ValueError("Not enough states for the desired number of electrons.");
    elif n_elec < 2:
        raise ValueError("Need at least two electrons to calculate J.")

    # Here we create a map between the ith spin-orbit state and the
    # corresponding explicit orbital and spin state.
    map_so_basis = np.zeros((n_se_so, 2));
    for idx in range(n_se_so):
        map_so_basis[idx, :] = [idx // 2, idx % 2]
    
    # Order the spin-orbital basis by spin then orbital
    sort_idx = np.lexsort((map_so_basis[:,0], map_so_basis[:,1]))
    map_so_basis = map_so_basis[sort_idx,:]

    # Get all possible state configurations and total number (will be cut
    # down a bit later on)
    state_configs = np.array(list(itertools.combinations(range(n_se_so), n_elec)))
    n_2q_states = qd.utils.nchoosek(n_se_so, n_elec)
    
    # Now decode the states found using nchoosek into our format where the
    # first K = n_elec indicies correspond to the orbital state and the last
    # K = n_elec indicies correspond to the spin state.
    # As an explicit example for a K = 3 case, consider the multi-electron 
    # spin-orbit state [4,2,3,0,0,1] which means:
    # 1st electron is in the 4th orbital state (idx=0) with spin down (idx=3)
    # 2nd electron is in the 2nd orbital state (idx=1) with spin down (idx=4)
    # 3rd electron is in the 3rd orbital state (idx=2) with spin up   (idx=5)
    vec_so_basis = np.zeros((n_2q_states, 2 * n_elec), dtype=int);
    for idx in range(n_2q_states):
        curr_config = state_configs[idx, :]
        curr_vec = np.zeros((2 * n_elec))
        
        for jdx in range(n_elec):
            curr_vec[jdx] = map_so_basis[curr_config[jdx], 0]
            curr_vec[jdx + n_elec] = map_so_basis[curr_config[jdx], 1]
                                                
        vec_so_basis[idx, :] = curr_vec
        
    # Rewrite each state to the following convention:
    # From left to right, first spin-down and then spin-up.  Within a spin
    # species from left to right, sort by ascending energy level (i.e.
    # orbital index i). This simplifies construction of the 2nd quantization
    # Hamiltonian
    for idx in range(n_2q_states):
        temp = vec_so_basis[idx, :]
        
        temp = np.reshape(temp, [n_elec, 2], order='F')
        
        sort_idx = np.lexsort((temp[:, 0], temp[:, 1]))
        temp = temp[sort_idx, :]
        
        vec_so_basis[idx, :] = np.reshape(temp, [2 * n_elec], order='F')

    # Now we want to truncate the spin subspace if desired
    # First calculate the possible spin subspaces
    possible_Sz = np.linspace(-n_elec/2, n_elec/2, n_elec+1)

    desired_sz = possible_Sz[spin_subspace]
    
    # Loop through each configuration and calcuate Sz.  If it is not
    # one of the desired Sz subspaces, then remove that basis vector.
    # Note that we loop backwards through the states here because we
    # remove rows from the basisVectors matrix which readjusts the
    # indices of the non-deleted rows.
    for idx in reversed(range(n_2q_states)):
        
        curr_spin_config = vec_so_basis[idx, n_elec:]
        curr_config_sz = 0
        
        for jdx in range(n_elec):
            # spin down
            if curr_spin_config[jdx] == 0:
                curr_config_sz -= 0.5
            # spin up
            else: 
                curr_config_sz += 0.5

        if not curr_config_sz in desired_sz:
            vec_so_basis = np.delete(vec_so_basis, (idx), axis=0)
                
    return vec_so_basis, map_so_basis


def build_second_quant_ham(n_elec: int, spin_subspace, n_se_orbs: int, 
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
        elemtns of the array. To use all spin subspaces, set spin_subspace='all'.
        The default is 'all'.
        n_se_orbs: int
            Number of single-electron orbitals basis states..
        se_energies: float 1d  array
            Corresponding single electron orbital basis state eigenenergies.
        se_cmes : double 2D array
            Coulomb matrix elements in the single electron basis.

    Returns
    -------
    H2ndQ : double 2D array
        Second quantization hamiltonian.

    '''
    
    se_energies = np.array(se_energies)
    if len(se_energies) != n_se_orbs:
        raise ValueError("The number of suppled single electron energies in"+
                         f" se_energies: {len(se_energies)} must be equal to"+
                         f" n_se_orbs: {n_se_orbs}.\n")
    
    vec_so_basis, map_so_basis = build_so_basis_vectors(n_elec, spin_subspace,
                                                        n_se_orbs)
    
    # Get number of 2nd quantization spin-orbital basis states
    n_2q_states = vec_so_basis.shape[0]

    # Initialize 2nd quantization Hamiltonian
    T = np.zeros((n_2q_states,n_2q_states), dtype=complex)
    Hc = np.zeros((n_2q_states,n_2q_states), dtype=complex)
                
    # The second quantization Hamiltonian has two terms: T + H_c.  T
    # describes all the single particle energies while
    # H_c describes all the direct and exchange electron interactions.
    
    # Let's first build the single-particle energy operator as it's the
    # easiest.
    # All elements lie only on the diagonal and each element is simply the
    # sum of single electron energies comprising the state
    # sum_j(\eps_j c_j^\dag c_j)
    # First get all of the energies
    
    for idx in range(n_2q_states):
        curr_se_orbs = vec_so_basis[idx, :n_elec]
        T[idx, idx] = sum(se_energies[curr_se_orbs])
    
    # Now that the T matrix is assembled, let's turn to the H_c term.
    for ndx in range(n_2q_states):
        for mdx in range(ndx + 1, n_2q_states):
            Hc[ndx, mdx] = __hc_helper(n_elec, ndx, mdx, n_se_orbs, se_cmes,
                                       vec_so_basis, map_so_basis)
            
    # Get lower triangular part of matrix
    Hc = Hc + Hc.conj().T

    # Get diagonal elements
    for ndx in range(n_2q_states):
        Hc[ndx, ndx] = __hc_helper(n_elec, ndx, ndx, n_se_orbs, se_cmes,
                                   vec_so_basis, map_so_basis)
    
    # Build the full Hamiltonian (and correct numerical errors by forcing
    # it to be symmetric)
    H2ndQ = T + Hc
    H2ndQ = (H2ndQ + H2ndQ.conj().T) / 2

    return H2ndQ

def __hc_helper(n_elec:int, ndx:int , mdx:int, n_se_orbs:int, se_cmes, 
                            vec_so_basis, map_so_basis):
    '''
    Helper function for constructing H_c term of second quantization hamiltonian

    Parameters
    ----------
    n_elec : int
        Number of electrons in the system.
    ndx : int
        Current many electron spin orbit basis state index for 1st electron.
    mdx : int
        Current many electron spin orbit basis state index for 2nd electron.
    n_se_orbs : int
        Number of single electron orbital basis states.
    se_cmes : double 2D array
        Coulomb Matrix Elements in the single electron basis.
    vec_so_basis : int 2D array
        Compilation of all many electron spin orbit basis states. First index
        corresponds to the ith state and the other indices are for the individual
        electron states for that given many electron state. Format is as follows:
        The first K = n_elec indicies correspond to the orbital state and the last
        K = n_elec indicies correspond to the spin state.
        As an explicit example for a K = 3 case, consider the multi-electron 
        spin-orbit state [4,2,3,0,0,1] which means:
        1st electron is in the 4th orbital state (idx=0) with spin down (idx=3)
        2nd electron is in the 2nd orbital state (idx=1) with spin down (idx=4)
        3rd electron is in the 3rd orbital state (idx=2) with spin up   (idx=5)
    map_so_basis : int 2D array
        A 2D array which maps the ith single electron spin orbit state (first
        index) to the corresponding single electron orbital and spin state.

    Returns
    -------
    hc_elem : double
        Calculated Coulomb Matrix Element <n|V|m>.

    '''
    # Get number of single electron spin orbital states
    n_se_so = map_so_basis.shape[0]

    hc_elem = 0
    
    # This function simply takes an input ket and applies the annihilation
    # operator corresponding to the n = [ind] spin-orbit state
    def __annihilation_helper(state, idx, map_so_basis):
        '''
        This function takes an input ket and applies the annihilation
        operator corresponding to the n = [idx] spin-orbit state

        Parameters
        ----------
        state : float 1D array
            Inputted ket vector state before annihilation.
        idx : int
            Index of the spin-orbit state to annihilate. See build_so_basis 
            function for the explanation of the index encoding rules. 
        map_so_basis: int 2D array
            Rows represent encodings of the spin-orbit states for the single
            electron spin and orbital states. Orbital states are in the 
            1st column, spin states are in the 2nd column. Ordered by spin,
            then by orbital.

        Returns
        -------
        state: float 1D array
            New spin-orbit many electron state.   
        '''

        annih_idx = map_so_basis[idx, :]
    
        for jdx in range(n_elec):
            # Check if the ith annihilation operator destroys any of the
            # states in the ket
            
            if state[jdx] == annih_idx[0] \
                and state[jdx + n_elec] == annih_idx[1]:
                # Edit the orbital and spin state so we know it was destroyed
                state[jdx], state[jdx + n_elec] = -1, -1
                
                # This will only happen once so break out of the loop
                break
            
        return state
  
    def __phase_helper(state):
        '''
        This function takes an inputted ket that has been modified 
        by annhilation operators (-1's introduced at the places where the
        electron was annihilated) and calculates the phase term
        caused by swapping of the fermionic operators
        during the annihilation applications.

        Parameters
        ----------
        state : 1D array
            State of interest.

        Returns
        -------
        phase : float
            Accumulated phase from repeated annihilation operations.

        '''

        # First find all the -1 indices in the state vector as these
        # correspond to annihilated electrons
        annih_ids = np.where(state == -1)[0]
        
        # Truncate the second half of these indices which correspond to spin
        # and not orbital.
        annih_ids = annih_ids[:(len(annih_ids) // 2)]
        
        # The remaining indices correspond to exactly how many swaps were
        # in order to apply each annihilation operator, so calculate the phase
        phase = (-1)**sum(annih_ids)
        
        return phase


    #*****************#
    # Loop over i > j #
    #*****************#
    for jdx in range(n_se_so - 1):
        # Initialize the bra state <ij|
        bra_orig = vec_so_basis[ndx, :]
        
        # jth annihilation operator
        bra_orig = __annihilation_helper(bra_orig, jdx, map_so_basis)
        
        # Used for the final two checks in the innermost loop (ldx)
        j_orb = map_so_basis[jdx, 0]
        j_spin = map_so_basis[jdx, 1]
        
        for idx in range(jdx + 1, n_se_so):
            # Refresh bra state for new loop
            bra = bra_orig
            
            # ith annihilation operator
            bra = __annihilation_helper(bra, idx, map_so_basis)

            # Now that we have the modified bra state, check that both
            # annihilation operators acted on it. If they both did not, then
            # they can be commuted so the one that did not act on the
            # vacuum state giving c|0> = 0.
            n_annih_applied = sum(bra == -1)
            
            # Divide by 2 because we modify both the orbital and spin part of 
            # the bra state
            if n_annih_applied // 2 != 2:
                continue
            
            # Now calculate the phase from applying these annihilation operators
            bra_phase = __phase_helper(bra)
            
            # Remove all annihilated electrons from our many-electron bra
            bra_trim = bra[bra != -1]
            
            # Used for the final two checks in the innermost loop (ldx)
            i_orb = map_so_basis[idx, 0]
            i_spin = map_so_basis[idx, 1]

            # Loop over k < l
            for kdx in range(n_se_so - 1):
                # Initialize the ket state |kl>
                ket_orig = vec_so_basis[mdx, :]
                
                # kth annihilation operator
                ket = __annihilation_helper(ket_orig, kdx, map_so_basis)
                
                # Used for the final two checks in the innermost loop (ldx)
                k_orb = map_so_basis[kdx, 0]
                k_spin = map_so_basis[kdx, 1]
                
                for ldx in range(kdx + 1, n_se_so):
                    # Refresh ket state for new loop
                    ket = ket_orig
                    
                    # Apply annihilation operators to ket state
                    # lth annihilation operator
                    ket = __annihilation_helper(ket, ldx, map_so_basis)
                    
                    # Now that we have the modified ket state, check that both
                    # annihilation operators acted on it.  If they both did not, then
                    # they can be commuted so the one that did not acts on the
                    # vacuum state giving c|0> = 0.
                    n_annih_applied = sum(ket == -1)
                    
                    # Divide by 2 because we modify both the orbital and spin 
                    # part of the ket state
                    if n_annih_applied // 2 != 2:
                        continue
                    
                    # Now calculate the phase from applying these annihilation
                    # operators
                    ket_phase = __phase_helper(ket)
                    
                    # Remove all annihilated electrons from our
                    # many-electron ket
                    ket_trim = ket[ket != -1]
                    
                    # SANITY CHECK: there should be nElec - 2 
                    # electrons remaining.
                    if (len(bra_trim) // 2 != (n_elec - 2) or
                            len(bra_trim) != len(ket_trim)):
                        raise ValueError('Incorrect number of electrons left '+
                                         'after application of bra/ket'+
                                         ' annhilation operators.')
                    
                    # Check that all the remaining electrons are the same for
                    # the bra and ket. Otherwise, the inner product is 0.
                    # Because of our ordering convention, after we removed the
                    # annihilated states, the arrays should match exactly.
                    # No need to worry about permutated vectors.
                    if not np.array_equal(bra_trim, ket_trim):
                        continue
                    
                    # Used for the final two checks in the innermost loop (ldx)
                    l_orb = map_so_basis[ldx, 0]
                    l_spin = map_so_basis[ldx, 1]
                                        
                    # Check that the spins for state pairs (i,l) and (j,k) 
                    # match
                    if i_spin == l_spin and j_spin == k_spin:
                        row_idx = int(j_orb * n_se_orbs + i_orb)
                        col_idx = int(k_orb * n_se_orbs + l_orb)
                                                
                        curr_cme = se_cmes[row_idx, col_idx]
                        
                        hc_elem += curr_cme * bra_phase * ket_phase

                    # Check that the spins for state pairs (i,k) and (j,l) 
                    # match
                    if i_spin == k_spin and j_spin == l_spin:
                        row_idx = int(j_orb * n_se_orbs + i_orb)
                        col_idx = int(l_orb * n_se_orbs + k_orb)
                        
                        curr_cme = se_cmes[row_idx, col_idx]
                        
                        hc_elem -= curr_cme * bra_phase * ket_phase
                
    return hc_elem
    
if __name__ == "__main__":
    
    build_second_quant_ham(3, [0, 1], 4, [15.5, 16.5, 17.5, 18.5])
        
        
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        