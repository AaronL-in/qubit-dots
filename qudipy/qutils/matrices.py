'''
Constant matrices used to define quantum gates
@author: hromecB, zmerino
'''
import os
import numpy as np

class Unitary:
    '''

    Initialize unitary class. Handles all things realted to
    creating/loading/modifying operator object files which contain various
    unitary operators in dictionary format. The Unitary class also contains hard
    coded unitary operators.

    '''

    ##### Operator Object Manipulation #####

    def __init__(self, operators=None, filename=None):
        '''

        Keyword Arguments
        -----------------
        operators: Dictionary
            Dictionary of unitary operators to be used to initialize an object.
            If left empty then the dictionary is going to be built from scratch
            or loaded from an operator object file from the working directory.
        filename: string
            String that indicates the name of the existing object file which
            contains a library of unitary operators.

        Returns
        -------
        None.

        '''
        # Ensure initial operator dictionary only contains unitary items with
        # the correct data structure, then assign to object
        if filename is None:
            if operators is None:
                # Empty dictionary
                self.operators = {}
            elif operators is not None:
                # Use user defined dictionary without existing object file
                self.check_ops(operators)
                self.operators = operators
        elif  filename is not None:
            
            # Append '.npz' if only the filename with out file type suffix
            if filename[-4:] != '.npz':
                filename = f'{filename}.npz'

            if operators is None:
                # Load dictionary object file with no user defined dictionary
                self.operators = self.load_ops(filename)
                self.check_ops(self.operators)
            elif operators is not None:
                # Use user defined dictionary and append existing object file
                self.operators = operators
                loaded_ops = self.load_ops(filename)
                self.add_operators(loaded_ops)
                self.check_ops(self.operators)

        # Default operator dictionary
        default_ops = {
                'PAULI_X': np.array([[0, 1], [1, 0]], dtype=complex),
                'PAULI_Y': np.array([[0,-1.0j], 
                    [1.0j, 0]],dtype=complex),
                'PAULI_Z': np.array([[1, 0], [0, -1]], dtype=complex),
                'PAULI_I': np.array([[1, 0], [0, 1]], dtype=complex)
            }    
        
        # Always have Pauli operators and the identity added. 
        # NOTE: this will overwrite dictionary values with same key name if 
        # loading a dictionary from an object file
        self.add_operators(default_ops)

    def check_ops(self, dict):
        '''
        
        Checks that an dictionary's operator items are in the correct format and
        unitary.   

        Parameters
        ----------
        dict : Dictionary
            Operator dictionary

        Returns
        -------
        None.

        '''
        if bool(dict) != False:
            # Check each item in the dictionary
            for key,val in dict.items():

                U = dict[key]
                Ustar = np.conjugate(np.transpose(U))

                # Check if array is square
                [N,M] = np.shape(U)

                # Ndarray is not a square matirx
                if N != M:
                    raise ValueError('Operator entry contains a non-sqaure'+
                        ' array of size [{},{}].'.format(N,M))

                # Create identity operator
                I = np.eye(N,N,dtype=complex)

                # Check if values are ndarrays
                if isinstance(val, np.ndarray) == False:
                    raise ValueError('The data type for item '+
                        '{} is not an ndarray, but {}.'.format(key,type(val)))
                
                # Check if array is complex object
                if np.iscomplexobj(val) == False:
                    raise ValueError('The object type for item '+
                        '{} is not complex.'.format(key))

                # Check it operator is unitary
                if (np.array_equal(np.matmul(U,Ustar),I) == False or
                    np.array_equal(np.matmul(Ustar,U),I) == False):
                    raise ValueError('Operator {} is not unitary.'.format(key))

    def save_ops(self, filename):
        '''
        
        Save the operator dictionary in the current working directory as an 
        object file for later use.

        Parameters
        ----------
        filename : String
            Name of operator dictionary file. If the file type suffix is not 
            included in the filename, it will be appended on to the filename
            string.

        Returns
        -------
        None.

        '''

        # Save operator object file to working directory
        np.savez(filename, **self.operators)
        print('Saved existing dictionary with unitary operators: {}.'.format(
            self.operators.keys()))
        
    def load_ops(self, filename):
        '''
        
        Load the operator object file for later use.

        Parameters
        ----------
        filename : String
            Name of the file with extension .npz for the operator object.

        Returns
        -------
        ops: Dictionary
            Dictionary containing unitary operators.

        '''
        # Load filename data object and convert to dictionary
        ops = dict(np.load(filename))

        print('Loaded existing dictionary with unitary operators: {}.'.format(
            ops.keys()))



        return ops

    def add_operators(self, new_ops):
        '''
        
        Add operator dictionary to an existing operator dictionary.

        Parameters
        ----------
        new_ops : Dictionary
            Dictionary of operators to add to exisiting dictionary.

        Returns
        -------
        None.

        '''
        print('Adding unitary operators: {}.'.format(new_ops.keys()))
        # Check the new operators
        self.check_ops(new_ops)
        # Merge existing dictionary with dictionary of new unitary operators
        self.operators = {**self.operators, **new_ops}

    def remove_operators(self, op_names):
        '''
        
        Remove operator items from existing operator dictionary.

        Parameters
        ----------
        op_names : List
            List of dictionary key strings that are to be removed from the
            existing dictionary.

        Returns
        -------
        None.

        '''
        print('Removing unitary operators: {}.'.format(op_names))
        for name in op_names:
            self.operators.pop(name)
    
    def construct(self, N, k, O):
        '''
        Creates matrix U_k of dimensions 2**N x 2**N for specified operator O
        
        Parameters
        ----------
        N: int
            Number of 1-qubit degrees of freedom in the operator.
        k: int
            Position of the Pauli matrix in the tensor product.
        O: Complex array
        
        Returns
        -------
        u_k: 2D complex array
            Matrix U_k of dimensions 2**N x 2**N 
        '''

        PAULI_I = np.array([[1, 0], [0, 1]], dtype=complex)

        if k == 1:
            return np.kron(O, np.eye(2 ** (N - 1)))
        u_k = PAULI_I
        for m in range(2, N + 1):
            if k is m:
                u_k = np.kron(u_k, O)
            else:
                u_k = np.kron(u_k, PAULI_I)
        else:
            return u_k

    ##### Hard Coded Operators #####

    # Ladder operators X_k ± i Y_k
    def sigma_plus(self, N, k):
        '''
        Defines a raising operators of the k-th qubit
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the raising operators in the tensor product.
        Returns
        -------
        : complex 2D array
            The raising operators X_k + i Y_k
        '''
        return (self.construct(N,k,self.operators['PAULI_X']) 
            + 1.0j * self.construct(N,k,self.operators['PAULI_Y']))

    def sigma_minus(self, N, k):
        '''
        Defines a lowering operators of the k-th qubit
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the raising operators in the tensor product.
        Returns
        -------
        : complex 2D array
            The lowering operators X_k - i Y_k
        '''
        return (self.construct(N,k,self.operators['PAULI_X']) 
            - 1.0j * self.construct(N,k,self.operators['PAULI_Y']))
        
    def e_up(self,N, k):
        '''
        Defines matrix that projects k-th qubit on the state |↑〉≡ |0〉
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the projection matrix in the tensor product.
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
            Matrix |0〉〈0|_k of dimensions 2**N x 2**N 
        '''
        return 0.5 * (self.unit(N) + self.construct(N,k,self.operators['PAULI_Z']))

    def e_down(self,N, k):
        '''
        Defines matrix that projects k-th qubit on the state |↓〉≡ |1〉
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the projection matrix in the tensor product.
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
            Matrix |1〉〈1|_k of dimensions 2**N x 2**N 
        '''
        return 0.5 * (self.unit(N) - self.construct(N,k,self.operators['PAULI_Z']))

    def unit(N):
        '''
        Defines unit matrix of dimensions 2**N x 2**N
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
        Unit matrix of dimensions 2**N x 2**N
        '''
        
        return np.eye((2 ** N), (2 ** N), dtype=complex)

    def cnot(self,N, ctrl, trgt):
        '''
        Defines a matrix for CNOT gate.
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        ctrl: int
            Index of the control qubit.
        trgt: int
            Index of the target qubit.
            
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for CNOT gate
        '''
        return (self.e_up(N, ctrl) + self.e_down(N, ctrl) 
            @ self.construct(N,trgt,self.operators['PAULI_X']))

    def swap(self,N, k1, k2):
        '''
        Defines SWAP gate matrix for the qubits with the indices k1, k2.
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k1, k2: int
            Indices of the qubits.
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for SWAP gate 
        '''
        return self.cnot(N, k1, k2) @ self.cnot(N, k2, k1) @ self.cnot(N, k1, k2)

    def sigma_product(self,N, k1, k2):
        '''
        Defines the dot product of two Pauli vectors.
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k1, k2: int
            Indices of the qubits.
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            The inner product \vec{sigma_k1} \cdot \vec{sigma_k2}
        '''

        return (self.construct(N, k1,self.operators['PAULI_X']) 
            @ self.construct(N, k2,self.operators['PAULI_X']) 
            + self.construct(N, k1,self.operators['PAULI_Y']) 
            @ self.construct(N, k2,self.operators['PAULI_Y']) 
            + self.construct(N, k1,self.operators['PAULI_Z']) 
            @ self.construct(N, k2,self.operators['PAULI_Z']))

    def rswap(self,N, k1, k2):
    
        '''
        Defines sqrt(SWAP) gate matrix for the qubits with the indices k1, k2.
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k1, k2: int
            Indices of the qubits.
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for SWAP gate 
        '''
        return (complex(0.25, -0.25) * self.sigma_product(N, k1, k2) 
            + complex(1.5, 0.5) * self.unit(N))

