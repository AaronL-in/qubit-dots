"""
Class used define general unitary operators

@author: zmerino
"""
import numpy as np

class Unitary:
    '''

    Inittialize unitary class. Handles all things realted to
    creating/loading/modifying operator object files which contain various
    unitary operators in dictionary format.

    '''

    def __init__(self,operators={}):

        '''

        Parameters
        ----------
        operators: Dictionary
            Dictionary of unitary operators to be used to initialize an object.
            If left empty then the dictionary is going to be built from scratch
            or loaded from an operator object file from the working directory.


        Returns
        -------
        None.

        '''
        # Ensure initial operator dictionary only contains unitary items with
        # the correct data structure, then assign to object
        self.check_ops(operators)
        self.operators = operators

    def check_ops(self,dict):
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
                    raise ValueError("Operator entry contains a non-sqaure"+
                        " array of size [{},{}].".format(N,M))

                # Create identity operator
                I = np.eye(N,N,dtype=complex)
                # Check if values are ndarrays
                if isinstance(val, np.ndarray) == False:
                    raise ValueError("The data type for item "+
                        "{} is not an ndarray, but {}.".format(key,type(val)))
                
                # Check if array is complex object
                if np.iscomplexobj(val) == False:
                    raise ValueError("The object type for item "+
                        "{} is not complex.".format(key))

                # Check it operator is unitary
                if (np.array_equal(np.matmul(U,Ustar),I) == False or
                    np.array_equal(np.matmul(Ustar,U),I) == False):
                    raise ValueError("Operator {} is not unitary.".format(key))

    def save_ops(self,filename):
        '''
        
        Save the operator dictionary in the current working directory as an 
        object file for later use.

        Parameters
        ----------
        filename : String
            Name of the file with extension .npz for the operator object.

        Returns
        -------
        None.

        '''
        # save operator object file to working directory
        np.savez(filename, **self.operators)
        
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
        print("Loaded existing dictionary with unitary operators: {}.".format(
            ops.keys()))
        return ops

    def add_operators(self,new_ops):
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
        print("Adding unitary operators: {}.".format(new_ops.keys()))
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
        print("Removing unitary operators: {}.".format(op_names))
        for name in op_names:
            self.operators.pop(name)