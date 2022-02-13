'''
Constant matrices used to define quantum gates
@author: hromecB, zmerino
'''
import inspect
import numpy as np

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

        # Append '.npz' if only the filename with out file type suffix
        if filename is not None:
            if filename[-4:] != '.npz':
                filename = f'{filename}.npz'

        # Set file name of object instance so multiple library objects
        # can be referenced
        self.filename = filename

        # Ensure initial operator dictionary only contains items with
        # the correct data structure, then assign to object
        if self.filename is None:
            if operators is None:
                # Empty dictionary
                self.lib = {}
            elif operators is not None:
                # Use user defined dictionary without existing object file
                self.check_ops(operators)
                self.lib = operators
        elif  self.filename is not None:
            if operators is None:
                # Load dictionary object file with no user defined dictionary
                self.lib = self.load_ops(self.filename, f_type)
                self.check_ops(self.lib)
            elif operators is not None:
                # Use user defined dictionary and append existing object file
                self.lib = operators
                loaded_ops = self.load_ops(self.filename, f_type)
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
        if self.filename is None:
            return f'Operator(lib_file=N/A, lib={self.lib.keys()})'
        else:
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

    def save_ops(self, filename=None):
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
        # Default name for operator library object file
        if filename is None:
            filename = 'Operator Library.npz'

        # Save operator object file to working directory
        np.savez(filename, **self.lib)
        print('Saved dictionary with operators, {}, to the file: {}.'.format(
            self.lib.keys(), filename))
        
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

    def coded_op(self, save, **kwargs):
        '''
        Manages arbitrarily defined operators that were hard coded i.e. saves
        the operator to the operator library if specified in the operator
        function call and only computes the operator if it does not exist in
        the operator library.
        ----------
        func : function
            An arbitrary function is sent.
        op_key : str
            Operator key name.            
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
        **kwargs : dictionary
            General structure: {'var1': val1, ..., 'varN': valN,
                'struct': stuct, 'func', func}
            Contains variable names and values used to construct the operator,
            as well as, functions to help constuct the operator and operator 
            library name.
        Returns
        -------
        : complex 2D array
            The computed or loaded operator
            
        '''

        # Define operator key name     
        curframe = inspect.currentframe()
        calframe = inspect.getouterframes(curframe, 2)
        method_name = calframe[1][3]

        # Append variable names and values to the operator name
        op_key = method_name
        for key, val in kwargs.items():
            if key not in ['struct', 'func']:
                op_key += kwargs['struct'](key, val)

        # If operator doesn's exist, save it if specified. If the operator does
        # exist then retrieve it from library
        if op_key in self.lib:
            # Retrieve operator
            return self.lib[op_key]
        else:
            # Compute operator for any specified function
            op = kwargs['func'](self)
            
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

    def PAULI_X(self, N, k, save=False):
        
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
        Returns
        -------
        : complex 2D array
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k_val = k

            return self.construct(N_val,k_val,self.lib['PAULI_X'])

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k': k, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)

    def PAULI_Y(self, N, k, save=False):
        
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
        Returns
        -------
        : complex 2D array
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k_val = k

            return self.construct(N_val,k_val,self.lib['PAULI_Y'])

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k': k, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)

    def PAULI_Z(self, N, k, save=False):
        
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
        Returns
        -------
        : complex 2D array
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k_val = k

            return self.construct(N_val,k_val,self.lib['PAULI_Z'])

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k': k, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)

    # Ladder operator X_k + i Y_k
    def SIGMA_PLUS(self, N, k, save=False):
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
        Returns
        -------
        : complex 2D array
            The raising operators X_k + i Y_k
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k_val = k

            return self.PAULI_X(N_val, k_val) + 1.0j * self.PAULI_Y(N_val, k_val)

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k': k, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)

    # Ladder operator X_k - i Y_k
    def SIGMA_MINUS(self, N, k, save=False):
        
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
        Returns
        -------
        : complex 2D array
            The lowering operators X_k - i Y_k
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k_val = k

            return self.PAULI_X(N_val, k_val) - 1.0j * self.PAULI_Y(N_val, k_val)

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k': k, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)

    def UNIT(self, N, save=False):
       
        '''
        Defines UNIT matrix of dimensions 2**N x 2**N
        
        Parameters
        ----------
        N : int
            Number of 1-qubit degrees of freedom in the operators.
        save : bool
            boolean indicator from original function call to save the operator
            to the library if true.
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
        Unit matrix of dimensions 2**N x 2**N
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N

            return np.eye((2 ** N_val), (2 ** N_val), dtype=complex)

        # Define variable dictionary for operator
        kwargs = {'N': N, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)
        
    def E_UP(self, N, k, save=False):
        
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
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
            Matrix |0〉〈0|_k of dimensions 2**N x 2**N 
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k_val = k

            return 0.5 * (self.UNIT(N) + self.PAULI_Z(N_val, k_val))

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k': k, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)
        
    def E_DOWN(self, N, k, save=False):
      
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
            
        Keyword Arguments
        -----------------
        None.
        
        Returns
        -------
        : 2D complex array
            Matrix |1〉〈1|_k of dimensions 2**N x 2**N 
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k_val = k

            return 0.5 * (self.UNIT(N) - self.PAULI_Z(N_val, k_val))

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k': k, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)

    def CNOT(self, N, ctrl, trgt, save=False):
       
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
            
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for CNOT gate
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func 
            return f'_{key}{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            ctrl_val = ctrl
            trgt_val = trgt

            return (self.E_UP(N_val, ctrl_val) + self.E_DOWN(N_val, ctrl_val) 
                @ self.PAULI_X(N_val, trgt_val))

        # Define variable dictionary for operator
        kwargs = {'N': N, 'ctrl': ctrl, 'trgt': trgt, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)
    
    def SWAP(self, N, k1, k2, save=False):
      
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
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for sqt(SWAP) gate 
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func
            if key == 'N': 
                return f'_{key}{val}'
            else:
                return f'_{key}_{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k1_val = k1
            k2_val = k2

            return (self.CNOT(N_val, k1_val, k2_val) 
                @ self.CNOT(N_val, k2_val, k1_val) 
                @ self.CNOT(N_val, k1_val, k2_val))

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k1': k1, 'k2': k2, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)
   
    def SIGMA_PRODUCT(self, N, k1, k2, save=False):
             
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
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            The inner product \vec{sigma_k1} \cdot \vec{sigma_k2}
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func
            if key == 'N': 
                return f'_{key}{val}'
            else:
                return f'_{key}_{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k1_val = k1
            k2_val = k2

            return (self.PAULI_X(N_val, k1_val) 
                @ self.PAULI_X(N_val, k2_val) 
                + self.PAULI_Y(N_val, k1_val) 
                @ self.PAULI_Y(N_val, k2_val) 
                + self.PAULI_Z(N_val, k1_val) 
                @ self.PAULI_Z(N_val, k2_val))

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k1': k1, 'k2': k2, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)

    def RSWAP(self, N, k1, k2, save=False):

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
        Keyword Arguments
        -----------------
        None.
        Returns
        -------
        : 2D complex array
            Matrix for SWAP gate 
        '''

        # Structure for appened names formed from the kwarg keys and values
        def struct(key, val):
            # Ignored: struct and func
            if key == 'N': 
                return f'_{key}{val}'
            else:
                return f'_{key}_{val}'

        # Define a function to construct the operator when needed
        def func(self):
            N_val = N
            k1_val = k1
            k2_val = k2

            return (complex(0.25, -0.25) * self.SIGMA_PRODUCT(N_val, k1_val, k2_val) 
            + complex(1.5, 0.5) * self.UNIT(N_val))

        # Define variable dictionary for operator
        kwargs = {'N': N, 'k1': k1, 'k2': k2, 'struct': struct, 'func': func}

        # Check if the operator exist or needs to be saved
        return self.coded_op(save, **kwargs)