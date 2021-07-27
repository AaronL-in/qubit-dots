"""
Class used define general unitary operators

@author: zmerino
"""
import os, sys
sys.path.append(os.path.dirname(os.getcwd() + '\\QuDiPy'))

import json as js
import numpy as np

class unitary:

    def __init__(self,operators={}):

        '''
        Parameters
        ----------


        Keyword Arguments
        -----------------


        Returns
        -------
        : 2D complex array
            Matrix for SWAP gate
        '''

        # Ensure initial operator dictionary only contains unitary items with
        # the correct data structure, then assign to object
        self.check_ops(operators)
        self.operators = operators


    def check_ops(self,dict):
            if bool(dict) != False:
                # Check each item in the dictionary
                for key,val in dict.items():

                    U = dict[key]
                    Ustar = np.conjugate(np.transpose(U))

                    # Check if array is square
                    [n,m] = np.shape(U)
                    if n == m:
                        # Ndarray represents a square matirx
                        N = n
                    else:
                        raise ValueError("Operator entry contains a non-sqaure"+
                            " array of size [{},{}].".format(n,m))

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
        # define original directory and qudipy module directory to save obj
        original_dir = os.getcwd()
        os.chdir(original_dir + '\\qudipy')

        np.savez(filename, **self.operators)
        
        # Reset working directory to original   
        os.chdir(original_dir)

    def load_ops(self, filename):
        # define original directory and qudipy module directory to save obj
        original_dir = os.getcwd()
        os.chdir(original_dir + '\\qudipy')

        # Load filename data object and convert to dictionary
        ops = dict(np.load(filename))
        print("Loaded existing dictionary with unitary operators: {}.".format(
            ops.keys()))
                
        # Reset working directory to original        
        os.chdir(original_dir)    
        return ops

    def add_operators(self,new_ops):
        print("Adding unitary operators: {}.".format(new_ops.keys()))
        # Check the new operators
        self.check_ops(new_ops)
        # Merge existing dictionary with dictionary of new unitary operators
        self.operators = {**self.operators, **new_ops}

    def remove_operators(self, op_names):
        print("Removing unitary operators: {}.".format(op_names))
        for name in op_names:
            self.operators.pop(name)

    

