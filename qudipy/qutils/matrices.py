'''
Constant matrices used to define quantum gates
@author: hromecB, zmerino
'''
import os
import inspect
import numpy as np
import json

class Operator:
    '''

    Initialize Operator class. Handles all things realted to
    creating/loading/modifying operator object files which contain various
    operators in dictionary format. The Operator class also contains hard
    coded operators.

    '''

    ##### Operator Object Manipulation #####

    def __init__(self, operators=None, filename=None, f_type=None):
        '''

        Keyword Arguments
        -----------------
        lib: Dictionary
            Dictionary of operators to be used to initialize an object.
            If left empty then the dictionary is going to be built from scratch
            or loaded from an operator object file from the working directory.
        filename: string
            String that indicates the name of the existing object file which
            contains a library of operators.

        Returns
        -------
        None.

        '''

        # Initialize object attribute for later use
        self.is_unitary = None

        # Ensure initial operator dictionary only contains items with
        # the correct data structure, then assign to object
        if filename is None:
            if operators is None:
                # Empty dictionary
                self.lib = {}
            elif operators is not None:
                # Use user defined dictionary without existing object file
                self.check_ops(operators)
                self.lib = operators
        elif  filename is not None:
            
            # Append '.npz' if only the filename with out file type suffix
            if filename[-4:] != '.npz':
                filename = f'{filename}.npz'

            # Set file name of object instance so multiple library objects
            # can be referenced
            self.filename = filename

            if operators is None:
                # Load dictionary object file with no user defined dictionary
                self.lib = self.load_ops(filename, f_type)
                self.check_ops(self.lib)
            elif operators is not None:
                # Use user defined dictionary and append existing object file
                self.lib = operators
                loaded_ops = self.load_ops(filename, f_type)
                self.add_operators(loaded_ops)
                self.check_ops(self.lib)

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

        # Check if the default operators and any additional operators are 
        # unitary and updated the is_unitary attribute.
        self.is_unitary = self.check_unitary(self.lib)

    # Overrides the [] operator for the class
    def __getitem__(self, key):
        return self.lib[key]

    # For addeding operator libraries to existing library       
    def __setitem__(self, lib):
        self.add_operators(lib)
        
    # For removing operator libraries from existing library    
    def __delitem__(self, op_names):
        self.remove_operators[op_names]

    # Create key method
    def keys(self):
        return self.lib.keys()

    def check_unitary(self, dict):
        '''
        
        Checks if a dictionary's operator items are all unitary.   

        Parameters
        ----------
        dict : Dictionary
            Operator dictionary

        Returns
        -------
        None.

        '''

        if bool(dict) != False:

            # Zero correcsponds to the assumption that the operator dictionary
            # under consideration consist of unitaries until proven otherwise.
            count = 0
            is_unitary = True
                
            # Check each item in the dictionary
            for key in dict.keys():
                
                U = dict[key]
                Ustar = np.conjugate(np.transpose(U))

                # Get operator array size
                N = np.shape(U)[0]

                # Create identity operator
                I = np.eye(N,N,dtype=complex)

                # Check if operator is not unitary
                if (np.array_equal(np.matmul(U,Ustar),I) == False or
                    np.array_equal(np.matmul(Ustar,U),I) == False):
                    count += 1

                if count > 0:
                    is_unitary = False
        else:
            is_unitary = None

        return is_unitary

    def check_ops(self, dict):
        '''
        
        Checks that a dictionary's operator items are valid data structure 
        types.   

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

                # Check if array is square
                [N,M] = np.shape(U)

                # Ndarray is not a square matirx
                if N != M:
                    raise ValueError('Operator entry contains a non-square'+
                        ' array of size [{},{}].'.format(N,M))

                # Check if values are ndarrays
                if isinstance(val, np.ndarray) == False:

                    # Try to convert to ndarray
                    try:
                        
                        print('Note: Data type of {} is not ndarray, but is: {}.'.format(key,type(val)))
                        
                        val = np.asarray(val)
                    except:
                        raise ValueError('Failed to convert {} to ndarray.'.format(key))
                
                # Check if array is complex object
                if np.iscomplexobj(val) == False:
                    
                    # Try to convert to complex object
                    try:

                        print('Note: Data type of first element for {} is not complex, but is: {}.'.format(key,type(val[0][0])))
                        val = val.astype(complex)
                    except:
                        raise ValueError('Failed to convert {} complex data type.'.format(key))

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
        np.savez(filename, **self.lib)
        print('Saved existing dictionary with operators: {}.'.format(
            self.lib.keys()))
        
    def load_ops(self, filename, f_type=None, disp=None):
        '''
        
        Load the operator object file for later use.

        Parameters
        ----------
        filename : String
            Name of the file with extension .npz for the operator object.

        Returns
        -------
        lib: Dictionary
            Dictionary containing operators.

        '''

        # Load filename data object and convert to dictionary
        lib = dict(np.load(filename))
        if disp is True:
            print('Loaded existing dictionary with operators: {}.'.format(
                lib.keys()))



        return lib

    def add_operators(self, new_ops, disp=None):
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
        if disp is True:
            print('Adding operators: {}.'.format(new_ops.keys()))

        # Check the new operators
        self.check_ops(new_ops)

        # If any non-unitary operator exist assign false
        if self.check_unitary(new_ops) is False or self.is_unitary is False:
            self.is_unitary = False

        # Merge existing dictionary with dictionary of new operators
        self.lib = {**self.lib, **new_ops}

    def remove_operators(self, op_names, disp=None):
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
        if disp is True:
            print('Removing unitary operators: {}.'.format(op_names))

        for name in op_names:
            self.lib.pop(name)

        # Check if updated dictionary contains only unitary operators
        self.is_unitary = self.check_unitary(self.lib)
    
    def construct(self, N, k, O, key_prefix=None):
        '''
        Creates matrix U_k of dimensions 2**N x 2**N for specified operator O
        
        Parameters
        ----------
        N: int
            Number of 1-qubit degrees of freedom in the operator.
        k: int
            Position of the Pauli matrix in the tensor product.
        O: 2D complex array
        key_prefix: string
            optional key prefix name argument. Key structure is key_prefix_N#K#.
        
        Returns
        -------
        u_k: 2D complex array
            Matrix U_k of dimensions 2**N x 2**N 
        '''
        O = O

        if k == 1:
            u_k = np.kron(O, np.eye(2 ** (N - 1)))
        else:    
            u_k = self.lib['PAULI_I']
            for m in range(2, N + 1):
                if k is m:
                    u_k = np.kron(u_k, O)
                else:
                    u_k = np.kron(u_k, self.lib['PAULI_I'])

        # update the operator library when a key name prefix is provided
        if key_prefix is not None:
            
            op_key = '{}_N{}K{}'.format(key_prefix,N,k)
            
            # Prevent duplicate operator entries
            if op_key not in self.lib:

                # construct dictionary item for operator
                operator_def = {
                        op_key: u_k
                }
                
                # Add operator to loaded dictionary
                self.add_operators(operator_def)
                # Save to operator library object
                self.save_ops(self.filename)

        return u_k

    # Use function to check if specific operator key name exists
    def op_exist(self, **kwargs):

        # Retrive method name of caller method
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)

        # Dictionary used to switch operator key naming convention
        name_style = {1: '{}_N{}K{}'.format(calframe[1][3], kwargs['N'], kwargs['k'])}
        
        # Operator key name 
        op_key  = name_style[kwargs['case']]

        print(op_key)
        
        # Prevent duplicate operator entries
        if op_key in self.lib:
            return True, op_key, self.lib[op_key]
        else:
            return False, op_key


    ##### Hard Coded Operators #####

    # Ladder operators X_k ± i Y_k
    def sigma_plus(self, N, k, save=None):
        '''
        Defines a raising operator of the k-th qubit
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

        # Search for operator with defined file name convention
        op_exist, op_name = self.op_exist(N=N, k=k, case=1)

        # If operator doesn's exist, save it if specified. If the operator does
        # exist then retrieve it from library
        if op_exist is False:
            if save is True:
                # Compute operator
                u_k = (self.construct(N,k,self.lib['PAULI_X']) 
                    + 1.0j * self.construct(N,k,self.lib['PAULI_Y']))
                    
                # Construct dictionary item for operator
                operator_def = {
                        op_name: u_k
                }
                
                # Add operator to loaded dictionary
                self.add_operators(operator_def)
                # Save to operator library object
                self.save_ops(self.filename)

                return u_k
            else:
                # Retrieve operator
                return self.lib[op_name]
        else:
            if op_exist is False:
                # Compute operator
                u_k = (self.construct(N,k,self.lib['PAULI_X']) 
                        + 1.0j * self.construct(N,k,self.lib['PAULI_Y']))
                   
                return u_k
            else:
                # Retrieve operator
                return self.lib[op_name]


    def sigma_minus(self, N, k):
        '''
        Defines a lowering operator of the k-th qubit
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


        print(self.op_exist(N=N, k=k, case=1))


        return (self.construct(N,k,self.lib['PAULI_X']) 
            - 1.0j * self.construct(N,k,self.lib['PAULI_Y']))
        
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
        return 0.5 * (self.unit(N) + self.construct(N,k,self.lib['PAULI_Z']))

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
        return 0.5 * (self.unit(N) - self.construct(N,k,self.lib['PAULI_Z']))

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
            @ self.construct(N,trgt,self.lib['PAULI_X']))

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
            Matrix for sqt(SWAP) gate 
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

        return (self.construct(N, k1,self.lib['PAULI_X']) 
            @ self.construct(N, k2,self.lib['PAULI_X']) 
            + self.construct(N, k1,self.lib['PAULI_Y']) 
            @ self.construct(N, k2,self.lib['PAULI_Y']) 
            + self.construct(N, k1,self.lib['PAULI_Z']) 
            @ self.construct(N, k2,self.lib['PAULI_Z']))

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

