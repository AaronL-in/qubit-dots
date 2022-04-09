"""
Class for a quantum circuit

@author: simba
"""
import numpy as np

class QuantumCircuit:
    
    def __init__(self, circuit_name, n_qubits, pulse_dict):
        '''
        Initialize QuantumCircuit object

        Parameters
        ----------
        circuit_name : string
            Name of quantum circuit.
        n_qubits : int
            Number of qubits in the circuit.
        pulse_dict : dictionary of ControlPulse objects
            Dictionary containing all control pulses that will be used in the
            quantum circuit.

        Returns
        -------
        None.

        '''
        
        # Name of circuit file
        self.name = circuit_name
        # Number of qubits in the circuit
        self.n_qubits = n_qubits
        # Loaded gates 
        self.gates = pulse_dict
        
        # The circuit sequence for this quantum circuit
        self.circuit_sequence = []
        
        # Index to track which gate in sequence we are on
        self.curr_gate_idx = 0
        
        # Flag to determine is every gate in the circuit has a correctly
        # specified ideal gate
        self.specified_all_ideal = True 
        # Flag to determine if the .qirc file that was loaded is comprised 
        # ONLY of ideal gates. Default assumes it is
        self.ideal_circuit = True
        
    def reset_circuit_index(self):
       '''
       Reset the current gate index for the circuit sequence back to the 
       begining of the circuit.

       Returns
       -------
       None.

       '''
       
       self.curr_gate_idx = 0
        
    def add_gate(self, gate_name, ideal_gate, used_qubits, position=None, mult_gates=False):
        '''
        This function adds a gate into the quantum circuit sequence.

        Parameters
        ----------
        gate_name : Anyof(string, tuple(string))
            Name of gate being added to sequence. If
            it is a tuple, then it contains the strings
            of the names of the multiple gates
            added simultaneously.
        ideal_gate: Anyof(string, tupleof(string), None)
            "IDEAL_GATE" if the gate is ideal, and 
            None if it is not an ideal gate.
        used_qubits : Anyof(int list, tuple(int list))
            Indices of qubits acted on by the gate(s) 
            where a tuple of lists indicates multiple 
            gates acting simultaneously.

        Keyword Arguments:
        -------------------
        position: Int, None, optional
            The positive integer greater than 0 that 
            specifies where the gate should be added to
            the circuit. If position is None, the gate will
            added onto the end.
        mult_gates: Bool, optional
            The predicate that affirms if there are
            multiple gates acting in the circut 
            simultaneously. Default is False.

        Returns
        -------
        None.

        '''
        def create_gate(gate_name, ideal_gate, used_qubits):
            '''
            Creates a list containing the gate name, ideal_gate
            and used qubits
            
            Parameters
            ----------
            gate_name : string
                Name of gate being added to sequence.
            ideal_gate: string
                "IDEAL_GATE" if the gate is ideal, and the
                name of the gate if it is not an ideal gate.
            used_qubits : int list or tuple(int list)
                Indices of qubits acted on by the gate
    
            Returns
            -------
            List
            '''
            # Check if a single qubit index was loaded or if it was a list.
            # If not a list, make it one.
            if not isinstance(used_qubits, list):
                used_qubits = [used_qubits]
            
            # Make sure used_qubits contains only ints
            used_qubits = [int(qubit_idx) for qubit_idx in used_qubits]
        
            # Check that the gate we are adding does not have an invalid qubit
            # index (i.e. outside of the allowable values)
            if (any(qubit_idx > self.n_qubits for qubit_idx in used_qubits) or
                any(qubit_idx < 1 for qubit_idx in used_qubits)):
                raise ValueError("Problem loading circuit file: " +
                                f"{self.name}.\n" +
                                f"Gate {gate_name} could not be loaded as the " +
                                f"affected qubit indices {used_qubits}\n are " +
                                " greater than the number of qubits in the circuit " +
                                f"({self.n_qubits}) or is <= 0.\n")
        
            # If the ideal_gate is None type, then there was an issue reading the
            # gate when the .ctrlp file was loaded.
            if ideal_gate is None:
                self.specified_all_ideal = False
        
            # Add the gate to the circuit sequence
            return [gate_name, ideal_gate, used_qubits]
        
        
        # Check if it is not more than one gate being applied.
        if mult_gates:
            # Parse over the list of lists containing the used qubits.
            # and add them to a tuple
            gates = ()
            # Split up gates into a list
            gate_name = list(map(lambda g: g.strip(" "), gate_name.split("|")))
            for i, u_qb in enumerate(used_qubits):
                if position is None:
                    gates = gates + tuple([create_gate(gate_name[i], ideal_gate[i], u_qb)])
                else:
                    try:
                        idx = int(postion) -1
                    except ValueError:
                        print("The keyword argument 'position' must be the gate number of \
                        the added gate.")
                    gates = gates[:idx] + \
                            tuple([create_gate(gate_name[i], ideal_gate[i], u_qb)])  + \
                            gates[idx:]
        
        # else, only one gate is being applied
        else:
            # Put the gate in the proper spot in the list
            if position is None:
                gates = create_gate(gate_name, ideal_gate, used_qubits)
            else:
                try:
                    idx = int(postion) -1
                except ValueError:
                    print("The keyword argument 'position' must be the gate number of \
                    the added gate.")
                gates = gates[:idx] + \
                        [create_gate(gate_name, ideal_gate, used_qubits)]  + \
                        gates[idx:]
            
        # Add the gate to the circuit sequence.
        self.circuit_sequence.append(gates)
        
        
    def get_next_gate(self):
        '''
        Get the next gate in the quantum circuit sequence. If no more exist, 
        return None.

        Returns
        -------
        next_gate : [string, int list]
            Returns a list containing the next gate name and the affected 
            qubits by the gate.

        '''
        
        try:
            next_gate = self.circuit_sequence[self.curr_gate_idx]
            self.curr_gate_idx += 1
        except IndexError:
            next_gate = None
        
        return next_gate
    
    def load_more_gates(self, pulse_dict):
        '''
        Adds more controlPulse objects to the gate dictionary.

        Parameters
        ----------
        pulse_dict : dict of controlPulse objects
            Dictionary containing all the pulse objects to be added.

        Returns
        -------
        None.

        '''
        
        for pulse_key, pulse_value in pulse_dict.items():
            self.gates[pulse_key] = pulse_value
        
    
    def print_ideal_circuit(self):
        '''
        Prints out an ascii display of the loaded circuit sequence for the user.
    
        Parameters
        ----------
        None.
    
        Returns
        -------
        None.
    
        '''
        
        # Check that every gate has an ideal gate specified
        if not self.specified_all_ideal:
            print(f'Cannot print ideal circuit for {self.name}.')
            print('Some or all of the gates in the circuit do not have an')
            print('ideal gate specified. Please check the .ctrlp or .qcirc')
            print('files for errors.')
            return
        
        # *********** HELPER FUNCTION ***********
        
        def format_circ_str(circ_str, curr_gate, gate_flag, mult_gates):
            '''
            @author: Madi
            Formats the circuit string for one gate and returns
            the circuit string

            Parameters:
            ------------
            circ_str: Listof(Str)
                The list of strings containing the circuit
                diagram
            curr_gate: Listof( ['GateName', 'IdealGate', [affected qubits] ] )
                The current gate that the circuit is on
            gate_flag: Int
                The integer that lets us know what type of gate we are 
                writing
            mult_gates: Bool
                The predicate determining if there are multiple gates 
                being written in the same instance

            Returns:
            -----------
            Listof(Str) that contains the circuit written up to
            and including the current gate 
            '''
            # Extract ideal gate and affected qubits
            # Multiple different gates acting simultaneously
            if mult_gates:
                ideal_gates = []
                aff_qubits = []
                set_of_qubits = []
                for gate in curr_gate:
                    ideal_gates += [gate[1]]
                    aff_qubits += [gate[2]]
                    set_of_qubits += gate[2]
                set_of_qubits = set(set_of_qubits)
            
            # One type of gate acting on one or more qubits
            else:
                ideal_gates = [curr_gate[1]]
                aff_qubits = [curr_gate[2]]
                set_of_qubits = set(curr_gate[2])
            
            # Create set of inactive qubits for later use
            set_of_inactive = set(np.arange(1, self.n_qubits+1, 1)) \
                            - set_of_qubits
            
            # Find starting length of the strings for later
            start_len = len(circ_str[0])
            
            # Create variable to store lengths of gates
            len_of_gates = []
            
            # Build the strings for a qubit affected by gate, nont affected by 
            # gate, and empty space between qubit lines
            
            # Initialize list to hold a dictionary of the string information
            info = []
            for ideal_gate in ideal_gates:
                gate_dict = {}
                
                if ideal_gate in ['H','I']:
                    gate_flag += [1]
                    gate_dict['used_str'] = ideal_gate + '-'
                    gate_dict['non_used_str'] = '--'
                    gate_dict['empty_space'] = '  '
                
                if ideal_gate[:2] in ['RX','RY','RZ']:
                    gate_flag += [1]
                    gate_dict['used_str'] = ideal_gate + '-'
                    gate_dict['non_used_str'] = '------'
                    gate_dict['empty_space'] = '      '
                
                if ideal_gate[:3] in ['ROT']: 
                    gate_flag += [1]
                    gate_dict['used_str'] = ideal_gate + '-'
                    gate_dict['non_used_str'] = '--------------'
                    gate_dict['empty_space'] = '              '
                
                # SWAP gates
                if ideal_gate in ['RSWAP','SWAP']:
                    gate_flag += [2]
                    gate_dict['used_str'] = ideal_gate + '-'
                    gate_dict['non_used_str'] = ''.join(['-']*(len(ideal_gate)+1))
                
                # CTRL gates        
                if ideal_gate[:4] == 'CTRL':
                    gate_flag += [3]
                    gate_dict['used_str'] = ideal_gate + '-'
                    gate_dict['ctrl_str'] = '--o---'
                    gate_dict['non_used_str'] = '------'
                
                # Add the gate dictionary into the information list
                info += [gate_dict]
            
            # Now append respective strings as appropriate for each qubit
            # Single qubit gate
            for i, flag in enumerate(gate_flag):
                
                if flag == 1:
                    # Step over each qubit to find the ones that the gate 
                    # acts on, then write the string.
                    for idx in range(1,self.n_qubits+1):
                        if idx in aff_qubits[i]:
                            circ_str[2*(idx-1)] += info[i]['used_str']
                        else:
                            # If the qubit is not being acted upon, add the length
                            # of the gate to len_of_gates for later formatting
                            length = len(info[i]['non_used_str'])
                            if length not in len_of_gates: len_of_gates += [length]
                
                if flag == 2:
                    # Step over each qubit to find the ones that the gate 
                    # acts on, then write the string.
                    for idx in range(1,self.n_qubits+1):
                        # Is qubit affected by gate
                        if idx in aff_qubits[i]:
                            circ_str[2*(idx-1)] += info[i]['used_str']
                        else:
                            # If the qubit is not being acted upon, add the length
                            # of the gate to len_of_gates for later formatting
                            length = len(info[i]['non_used_str'])
                            if length not in len_of_gates: len_of_gates += [length]
                            
                    # Now fill in the empty spaces
                    for idx in range(1,self.n_qubits):
                        if idx in range(min(aff_qubits[i]),max(aff_qubits[i])):
                            if ideal_gate == 'SWAP':
                                circ_str[2*idx-1] += '  |  '
                            elif ideal_gate == 'RSWAP':
                                circ_str[2*idx-1] += '  |   '
                
                if flag == 3:
                    # Step over each qubit to find the ones that the gate 
                    # acts on, then write the string.
                    for idx in range(1,self.n_qubits+1):
                        # First qubit indices are always the ctrl qubits
                        if idx in aff_qubits[i][:-1]:
                            circ_str[2*(idx-1)] += info[i]['ctrl_str']
                        
                        # Last qubit index is always the target qubit
                        elif idx == aff_qubits[i][-1]:
                            circ_str[2*(idx-1)] += info[i]['used_str']
                        
                        else:
                            # If the qubit is not being acted upon, add the length
                            # of the gate to len_of_gates for later formatting
                            length = len(info[i]['non_used_str'])
                            if length not in len_of_gates: len_of_gates += [length]
                            
                    # Now fill in the empty spaces
                    for idx in range(1,self.n_qubits):
                        if idx in range(min(aff_qubits[i]),max(aff_qubits[i])):
                            circ_str[2*idx-1] += '  |   '

                
            # Now make sure that every element in the circ_str list is 
            # of the same length.
                
            # Find the longest gate name
            max_len = max(len_of_gates) + start_len + 1
                
            # Step over each element of the list to make sure it is of
            # max_len, if not, add the correct number of characters to it.
            # Note: this part also adds 'padding' of one '-' or ' ' character.
            for i, s in enumerate(circ_str):
                str_len = len(s)
                if i % 2 == 0:
                    if str_len != max_len:
                        circ_str[i] += ''.join(['-']*(max_len - str_len))
                else:
                    if str_len != max_len:
                        circ_str[i] += ''.join([' ']*(max_len - str_len))
            
            # Return the circ_str
            return circ_str    


        # Initialize the circuit to print
        circ_str = []
        for idx in range(self.n_qubits):
            circ_str.append('Q' + str(idx+1) + ' --')
            if idx != self.n_qubits:
                if idx < 10:
                    circ_str.append('     ')
                else:
                    circ_str.append('      ')
        
        # Each odd idx in circ_str corresponds to a qubit in the circuit
        # Each even idx correspond to gaps between qubit lines.
        
        # Store the current gate index to change back to later
        initial_gate_idx = self.curr_gate_idx
        # Now reset index
        self.curr_gate_idx = 0
        
        # Now loop through each gate in circuit and add the respective strings
        curr_gate = 0
        gate_flag = []
        curr_gate = self.get_next_gate()
        
        # Go through each gate and format the circ_str
        while curr_gate is not None:
            
            # Consider the case with simultaneous different gates acting
            # on the circuit at once
            mult_gates = False
            if isinstance(curr_gate, tuple):
                mult_gates = True

            # Write the circuit string
            circ_str = format_circ_str(circ_str, curr_gate, gate_flag, mult_gates)
            # Reset things for next loop        
            curr_gate = self.get_next_gate()
            gate_flag = []

        # Print the circuit
        print(f'Ideal circuit: {self.name}\n')
        for idx in range(len(circ_str)):
            print(circ_str[idx])
            
        self.curr_gate_idx = initial_gate_idx
        
        
        