"""
Quantum utility math functions

@author: simba
"""

import numpy as np

def find_overlap_matrix(gparams, basis_1_wfs, basis_2_wfs):
    '''
    Calculates the overlap matrix S between two sets of basis vectors. The
    size of S is NxM where N = size(basis 1) and M = size(basis 2).

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid information.
    basis_1_wfs : array
        Set of basis states where the first index in the array corresponds to 
        the ith state. The state must be a meshgrid if a 2D grid is used.
    basis_2_wfs : array
        Set of basis states where the first index in the array corresponds to 
        the ith state. The state must be a meshgrid if a 2D grid is used.

    Returns
    -------
    S_matrix : array
        Overlap matrix between basis 1 and basis 2.

    '''
    
    num_basis_1 = basis_1_wfs.shape[0]
    num_basis_2 = basis_2_wfs.shape[0]
    
    S_matrix = np.zeros((num_basis_2, num_basis_1), dtype=complex)
    
    for j_idx in range(num_basis_1):
        wf_1 = basis_1_wfs[j_idx]
        
        for i_idx in range(num_basis_2):
            wf_2 = basis_2_wfs[i_idx]
            
            # S_ij = <b2_i|b1_j>
            S_matrix[i_idx,j_idx] = inner_prod(gparams, wf_2, wf_1)

    return S_matrix

def inner_prod(gparams, wf1, wf2):
    '''
    Evaluates the inner product between two complex wavefunctions.

    Parameters
    ----------
    gparams : GridParameters object
        Contains grid and potential information.
    wf1 : complex array
        'Bra' wavefunction for the inner product. If grid is 2D, then the 
        array should be in meshgrid format.
    wf2 : complex array
        'Ket' wavefunction for the inner product. If grid is 2D, then the 
        array should be in meshgrid format.

    Returns
    -------
    inn_prod : complex float
        The inner product between the two inputted wavefunctions.

    '''
    
    if gparams.grid_type == '1D':
        inn_prod = np.trapz(wf1.conj()*wf2, x=gparams.x)
    elif gparams.grid_type == '2D':
        inn_prod = np.trapz(np.trapz(wf1.conj()*wf2, x=gparams.x, axis=1),
                            gparams.y)
            
    return inn_prod

def expectation_value(gparams, wf, operator):
    '''
    Evaluates the expectation value for an observable of a wavefunction.

    Parameters
    ----------
    gparams : GridParameters class
        Contains grid and potential information.
    wf : complex array
        Wavefunction for the expectation value. If grid is 2D, then the 
        array should be in meshgrid format.
    operator : complex array
        Matrix representation of the operator for the expectation value.
        
    Keyword Arguments
    ----------------
    None.

    Returns
    -------
    exp_val : complex float
        The expectation value for the observable of a wavefunction.

    '''
    
    if gparams.grid_type == '1D':
        exp_val = np.trapz(np.multiply(wf.conj(), np.matmul(operator, wf)), 
                           x=gparams.x)
    elif gparams.grid_type == '2D':
        exp_val = np.trapz(np.trapz(np.multiply(wf.conj(), np.matmul(operator, wf)), 
                           x=gparams.x, axis=1), gparams.y)
            
    return exp_val

def matrix_element(gparams, wf1, wf2, operator):
    '''
    Evaluates the matrix element between two complex wavefunctions.

    Parameters
    ----------
    gparams : GridParameters class
        Contains grid and potential information.
    wf1 : complex array
        'Bra' wavefunction for the matrix element. If grid is 2D, then the 
        array should be in meshgrid format.
    wf2 : complex array
        'Bra' wavefunction for the matrix element. If grid is 2D, then the 
        array should be in meshgrid format.
    operator : complex array
        Matrix representation of the operator for the matrix element.
        
    Keyword Arguments
    ----------------
    None.

    Returns
    -------
    mtx_elem : complex float
        The matrix element between two complex wavefunctions.

    '''
    
    if gparams.grid_type == '1D':
        mtx_elem = np.trapz(np.multiply(wf1.conj(), np.matmul(operator, wf2)), 
                           x=gparams.x)
    elif gparams.grid_type == '2D':
        mtx_elem = np.trapz(np.trapz(np.multiply(wf1.conj(), np.matmul(operator, wf2)), 
                           x=gparams.x, axis=1), gparams.y)
            
    return mtx_elem

def project_up(rho, elem):
    """
    Projects the system density matrix onto the spin-up state of 
    the specified electron(s) (without renormalizing it)
    
    Parameters
    ----------
    rho : 2D complex array
        Matrix of the dimensions 2**N x 2**N (density matrix in our case).
    elem : int / iterable of ints
        Number(s) of the electron(s) whose state(s) is to project the system 
        density matrix on.
        
    Keyword Arguments
    ----------------
    None.
    
    Returns
    -------
    projected_rho : complex 2D array
        new density matrix (non-renormalized) of the dimensions 
        2**m x 2**m, where m = N - dim(elem) 

    """
    dim = rho.shape[0]    #dimension of the matrix
    N = int(math.log2(rho.shape[0]))   
            #number of qubits encoded by density matrix
    
    if isinstance(elem, int):
        
        projected_rho=[]
            #the idea behind bitwise shifts is that every matrix in tensor product
            #doubles the dimension of the resulting matrix, i. e. adds one binary 
            #digit to the number. It follows from the definition of Kronecker
            #product that the most significant bit defines the element of the
            #leftmost matrix in Kronecker product, and vice versa
        for i in range(0, dim):
            if (i >> (N - elem)) % 2 == 0:
                var = []
                for j in range(0, dim):
                    if (j >> (N - elem)) % 2 == 0:
                        var.append(rho[i][j])
                projected_rho.append(var)
            else:
                continue
    
        return  np.array(projected_rho)
    
    if isinstance(elem, (tuple, list, set) ):
        projected_rho = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            # consecutive application of the project_up operations 
        for el in elems_sorted:
            projected_rho = project_up(projected_rho, el)
        return projected_rho
    
    raise ValueError("Qubits that are traced out should be defined by a  \
                        single int number or an iterable of ints. Try again")

def project_down(rho, elem):
    """
    Projects the system density matrix onto the spin-down state 
    of the specified electron(s) (without renormalizing it).
    
    Parameters
    ----------
    rho : 2D array
        Matrix of the dimensions 2**N x 2**N (density matrix in our case).
    elem : int / iterable of ints
        Number(s) of the electron(s) whose state(s) is to project the system 
        density matrix on.
        
    Keyword Arguments
    ----------------
    None.
    
    Returns
    -------
    projected_rho : complex 2D array
        new density matrix (non-renormalized) of the dimensions 
        2**m x 2**m, where m = N - dim(elem) 
    """

    dim = rho.shape[0]    #dimension of the matrix
    N = int(math.log2(rho.shape[0]))   
            #number of qubits encoded by density matrix
    
    if isinstance(elem, int):
        
        projected_rho = []
            #the idea behind bitwise shifts is that every matrix in tensor product
            #doubles the dimension of the resulting matrix, i. e. adds one binary 
            #digit to the number. It follows from the definition of Kronecker
            #product that the most significant bit defines the element of the
            #leftmost matrix in Kronecker product, and vice versa
        for i in range(0, dim):
            if (i >> (N - elem))%2 == 1:
                var = []
                for j in range(0, dim):
                    if (j >> (N - elem)) % 2 == 1:
                        var.append(rho[i][j])
                projected_rho.append(var)
            else:
                continue
    
        return np.array(projected_rho)
    
    if isinstance(elem, (tuple,list,set)):
        projected_rho = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            #consecutive application of the project_down operations 
        for el in elems_sorted:
            projected_rho = project_down(projected_rho, el)
        return projected_rho

    raise ValueError("Qubits that are traced out should be defined by a  \
                        single int number or an iterable of ints. Try again")


def partial_trace(rho, elem):
    """
    Finds partial trace with respect to the specified qubit(s)
    
    Parameters
    ----------
    rho : 2D array
        matrix of the dimensions 2**N x 2**N (density matrix in our case)
    elem : int / iterable of ints
        number(s) of the electron(s) whose state(s) is/are averaged out

    Returns
    -------
    traced_rho: complex 2D array
        new density matrix of the dimensions 2**m x 2**m, m = N - dim(elem) 

    """
    if isinstance(elem, int):
        return project_up(rho, elem) + project_down(rho, elem)
    
    if isinstance(elem, (tuple, set, list)):
        traced_rho = rho.copy()
        elems_sorted = sorted(elem, reverse=True)
            #iterable sorted in the reversed order; necessary for the correct 
            # consecutive application of the project_up operations 
        for el in elems_sorted:
            traced_rho = partial_trace(traced_rho, el)
        return traced_rho
    
    #error with the input 
    raise ValueError("Qubits that are traced out should be defined by a  \
                        single int number or an iterable of ints. Try again")

def matrix_sqrt(A):
    """
    Calculates a square root of the specified matrix.

    Parameters
    ----------
    A : numpy array
        matrix to calculate the square root of

    Returns
    -------
    sqrt_A: complex 2D array
        square root of the matrix

    """
    w,v = la.eig(A)
    sqrt_A = v @ np.diag(np.sqrt((1 + 0j)*w)) @ la.inv(v)
    return sqrt_A
    


def partial_trace_general(rho, dim, traced_subsystem):
    '''
    This code takes the partial trace of a density matrix.  It is adapted from
    the open-source TrX file written by Toby Cubitt for MATLAB.
    
    TODO: Add code for ket states as well.. I didn't understand how the MATLAB
    code was working.

    Parameters
    ----------
    rho : complex 2D array
        A 2D array describing a density matrix.
    dim : 1D array
        An array of the dimensions of each subsystem for the whole system for
        psi. [2,4,2] corresponds to a system of size 2x4x2 with 3 subsystems.
    sys : 1D array
        An array of the subsystems to trace out. For dim=[2,4,2], sys=[1,3] 
        would trace out the first and third subsystems leaving only a subspace
        with size 4.

    Returns
    -------
    traced_rho : complex 2D array
        The resulting rho after taking the partial trace.
        
    '''
    # Convert inputs to numpy arrays if not already inputted as such.
    rho = np.array(rho)
    sys = np.array(sys)
    dim = np.array(dim)
    
    # Check inputs
    if any(sys > len(dim)-1) or any(sys < 0):
        print(sys > len(dim))
        print(any(sys < 0))
        # Error with sys variable
        raise ValueError("Invalid subsytem in sys.")
    if ((len(dim) == 1 and np.mod(len(rho)/dim,1) != 0) 
        and len(rho) != np.prod(dim)):
        # Error with dim or psi variables
        raise ValueError("Size of rho inconsistent with dim.")
    
    # Get rid of any singleton dimensions in dim
    sys = np.setdiff1d(sys, np.argwhere(dim == 1))
    dim = np.concatenate([dim[idx] for idx in np.argwhere(dim != 1)])
    
    # Number of subsystems
    n = len(dim)
    dim_reversed = dim[::-1]
    subsystems_keep = np.array([idx for idx in np.linspace(0,n-1,n) 
                                if idx not in sys])
    # Dimension of psi to trace out
    dim_trace = dim[sys].prod()
    # Dimension of psi leftover after partial trace
    dim_keep = len(rho)/dim_trace
    
    # Reshape density matrix into tensor with one row and one column index for
    # each subsystem, permute traced subsytem indices to the end, reshape
    # again so that first two indices are row and column multi-indices for
    # kept subsystems and third index is a flatted index for traced subsytems,
    # then sum third index over "diagonal" entries.
    perm = n-1 - np.concatenate([subsystems_keep[::-1], subsystems_keep[::-1]-n,
                               sys, sys-n])
    rho1 = np.reshape(rho, np.concatenate([dim_reversed, dim_reversed]),
                      order='F')
    rho2 = np.transpose(rho1, perm.astype(int))
    rho = np.reshape(rho2, np.array([dim_keep, dim_keep, dim_trace**2],
                                    dtype=np.uint), order='F')
    traced_rho = np.sum(rho[:,:,range(0,dim_trace**2,dim_trace+1)], axis=2)
            
    return traced_rho












