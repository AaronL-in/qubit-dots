
def main():
    import os, sys
    sys.path.append(os.path.dirname(os.getcwd() + '\\QuDiPy'))
    original_dir = os.getcwd()

    import qudipy.qutils.unitaries as qunit

    import numpy as np
    import matplotlib.pyplot as plt
    
    init_ops = {
        'PAULI_X': np.array([[0, 1], [1, 0]], dtype=complex),
        'PAULI_Y': np.array([[0, complex(-0.0, -1.0)], 
            [complex(0.0, 1.0), 0]],dtype=complex),
        'PAULI_Z': np.array([[1, 0], [0, -1]], dtype=complex),
        'PAULI_I': np.array([[1, 0], [0, 1]], dtype=complex)
    }

    # Initialize unitary object
    # ops = qunit.unitary(init_ops)
    ops = qunit.unitary()
    
    filename = 'Unitary Operators.npz'
    ops.save_ops(filename)
    # define filename for unitary operators object
    print(filename)
    # save operators object
    
    # load dictionary of operators from unitary object
    ops.operators = ops.load_ops(filename)

    # For error checks
    # tutorial_ops = {
    #     'unitary1':  np.array([[1, 0], [0, 1]], dtype=complex),
    #     'unitary2':  np.array([[2, 0], [0, 2]], dtype=complex),
    #     'not-square':  np.array([[1, 0, 0], [0, 1, 0]], dtype=complex),
    #     'not-complex':  np.array([[1, 0], [0, 1]], dtype=int),
    #     'not-array':  [[-1, 0], [0, -1]],
    #     'not-unitary':  np.array([[1, 0], [0, 2]], dtype=complex)
    # }

    tutorial_ops = {
        'unitary1':  np.array([[1, 0], [0, 1]], dtype=complex),
        'unitary2':  0.5*np.array([[complex(1.0, -1.0), complex(1.0, 1.0)], 
            [complex(1.0, 1.0), complex(1.0, -1.0)]], dtype=complex)
    }

    # Attempt to add new operators to exist operators dictionary
    ops.add_operators(tutorial_ops)

    print(ops.operators.keys())

    op_names = ['unitary1','unitary2']

    ops.remove_operators(op_names)
    
    print(ops.operators.keys())



main()

# if __name__ == "__main__":
#     import cProfile
#     cProfile.run('main()', "output.dat")

#     import pstats
#     from pstats import SortKey

#     with open("output_time.txt", "w") as f:
#         p = pstats.Stats("output.dat", stream=f)
#         p.sort_stats("time").print_stats()     
        
#     with open("output_calls.txt", "w") as f:
#         p = pstats.Stats("output.dat", stream=f)
#         p.sort_stats("calls").print_stats()        
           