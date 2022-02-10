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

    # For printing human and computer readible object instance information
    def __repr__(self):
        return f'Operator(lib_file={self.filename}, lib={self.lib.keys()})'

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
        
    def load_ops(self, filename, f_type=None, disp=False):
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
        if disp:
            print('Loaded existing dictionary with operators: {}.'.format(
                lib.keys()))



        return lib

    def add_operators(self, new_ops, disp=False):
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
        if disp:
            print('Adding operators: {}.'.format(new_ops.keys()))

        # Check the new operators
        self.check_ops(new_ops)

        # If any non-unitary operator exist assign false
        if self.check_unitary(new_ops) is False or self.is_unitary is False:
            self.is_unitary = False

        # Merge existing dictionary with dictionary of new operators
        self.lib = {**self.lib, **new_ops}

    def remove_operators(self, op_names, disp=False):
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
        if disp:
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

    def man_coded_ops(self, func, save, case='0 input', **kwargs):
        '''
        Manages arbitrarily defined operators that were hard coded i.e. saves
        the operator to the operator library if specified in the operator
        function call and only computes the operator if it does not exist in
        the operator library.
        ----------
        func : function
            An arbitrary function is sent.
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the raising operators in the tensor product.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
        Returns
        -------
        : complex 2D array
            The computed or loaded operator
            
        '''

        # Retrieve method name of the caller method
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        method_name = calframe[1][3]

        # Nested function to handle variable name changes for operator library
        # names
        def var_name(method_name, **kwargs):
            if method_name == 'cnot':
                return '{}_N{}ctrl{}trgt{}'.format(method_name, kwargs['arg0'], 
                        kwargs['arg1'], kwargs['arg2'])
            else:
                return '{}_N{}k1_{}k2_{}'.format(method_name, kwargs['arg0'], 
                        kwargs['arg1'], kwargs['arg2'])

        # Dictionary used to switch operator key naming conventions
        name_style = {'0 input': '{}_N{}'.format(method_name, kwargs['arg0']),
                    '1 input': '{}_N{}k{}'.format(method_name, kwargs['arg0'], kwargs['arg1']),
                    '2 input': var_name(method_name, **kwargs)}

        # Operator key name 
        op_key  = name_style[case]
        
        # Prevent duplicate operator entries
        if op_key in self.lib:
            op_exist =  True
        else:
            op_exist =  False

        # If operator doesn's exist, save it if specified. If the operator does
        # exist then retrieve it from library
        if op_exist:
            # Retrieve operator
            return self.lib[op_key]
        else:
            # Compute operator for any specified function
            op = func(self)

            if save:
                # Construct dictionary item for operator
                operator_def = {
                        op_key: op
                }
                
                # Add operator to loaded dictionary
                self.add_operators(operator_def)
                # Save to operator library object
                self.save_ops(self.filename)

                return op

    ##### Hard Coded Operators #####

    # Ladder operators X_k ± i Y_k
    def sigma_plus(self, N, k, save=False, case='1 input'):
        '''
        Defines a raising operator of the k-th qubit
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the raising operators in the tensor product.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
        Returns
        -------
        : complex 2D array
            The raising operators X_k + i Y_k
        '''
        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':k, 'arg2':None}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            k = kwargs['arg1']

            return (self.construct(N,k,self.lib['PAULI_X']) 
                + 1.0j * self.construct(N,k,self.lib['PAULI_Y']))

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)

    def sigma_minus(self, N, k, save=False, case='1 input'):
        
        '''
        Defines a lowering operator of the k-th qubit
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the raising operators in the tensor product.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
        Returns
        -------
        : complex 2D array
            The lowering operators X_k - i Y_k
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':k, 'arg2':None}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            k = kwargs['arg1']

            return (self.construct(N,k,self.lib['PAULI_X']) 
            - 1.0j * self.construct(N,k,self.lib['PAULI_Y']))

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)
        
    def e_up(self, N, k, save=False, case='1 input'):
        
        '''
        Defines matrix that projects k-th qubit on the state |↑〉≡ |0〉
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the projection matrix in the tensor product.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
            Matrix |0〉〈0|_k of dimensions 2**N x 2**N 
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':k, 'arg2':None}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            k = kwargs['arg1']

            return 0.5 * (self.unit(N)+ self.construct(N,k,self.lib['PAULI_Z']))

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)
        
    def e_down(self, N, k, save=False, case='1 input'):
      
        '''
        Defines matrix that projects k-th qubit on the state |↓〉≡ |1〉
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k : int
            Position of the projection matrix in the tensor product.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
            Matrix |1〉〈1|_k of dimensions 2**N x 2**N 
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':k, 'arg2':None}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            k = kwargs['arg1']

            return 0.5 * (self.unit(N) - self.construct(N,k,self.lib['PAULI_Z']))

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)

    def unit(self, N, save=False):
       
        '''
        Defines unit matrix of dimensions 2**N x 2**N
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
        Unit matrix of dimensions 2**N x 2**N
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':None, 'arg2':None}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']

            return np.eye((2 ** N), (2 ** N), dtype=complex)

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, **kwargs)

    def cnot(self, N, ctrl, trgt, save=False, case='2 input'):
       
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
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
            
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for CNOT gate
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':ctrl, 'arg2':trgt}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            ctrl = kwargs['arg1']
            trgt = kwargs['arg2']

            return (self.e_up(N, ctrl) + self.e_down(N, ctrl) 
            @ self.construct(N,trgt,self.lib['PAULI_X']))

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)
    
    def swap(self, N, k1, k2, save=False, case='2 input'):
      
        '''
        Defines SWAP gate matrix for the qubits with the indices k1, k2.
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k1, k2: int
            Indices of the qubits.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for sqt(SWAP) gate 
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':k1, 'arg2':k2}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            k1 = kwargs['arg1']
            k2 = kwargs['arg2']

            return self.cnot(N, k1, k2) @ self.cnot(N, k2, k1) @ self.cnot(N, k1, k2)

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)
   
    def sigma_product(self, N, k1, k2, save=False, case='2 input'):
             
        '''
        Defines the dot product of two Pauli vectors.
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k1, k2: int
            Indices of the qubits.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            The inner product \vec{sigma_k1} \cdot \vec{sigma_k2}
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':k1, 'arg2':k2}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            k1 = kwargs['arg1']
            k2 = kwargs['arg2']

            return (self.construct(N, k1,self.lib['PAULI_X']) 
                @ self.construct(N, k2,self.lib['PAULI_X']) 
                + self.construct(N, k1,self.lib['PAULI_Y']) 
                @ self.construct(N, k2,self.lib['PAULI_Y']) 
                + self.construct(N, k1,self.lib['PAULI_Z']) 
                @ self.construct(N, k2,self.lib['PAULI_Z']))

        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)

    def rswap(self, N, k1, k2, save=False, case='2 input'):

        '''
        Defines sqrt(SWAP) gate matrix for the qubits with the indices k1, k2.
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        k1, k2: int
            Indices of the qubits.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        case : string
            Is a string that specifies the naming style for the operator's key
            label. The default is of the form <operator name>_N#k#.
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for SWAP gate 
        '''

        # Define variable dictionary for key name styles
        kwargs = {'arg0': N, 'arg1':k1, 'arg2':k2}

        # Define a function to construct the operator when needed
        def func(self):
            N = kwargs['arg0']
            k1 = kwargs['arg1']
            k2 = kwargs['arg2']

            return (complex(0.25, -0.25) * self.sigma_product(N, k1, k2) 
            + complex(1.5, 0.5) * self.unit(N))


        # Check if the operator exist or needs to be save
        return self.man_coded_ops(func, save, case, **kwargs)

