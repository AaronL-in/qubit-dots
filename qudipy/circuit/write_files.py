'''
Module to write .ctrlp files and .qcirc files 
from the respective ContrsolPulse and QuantumCircuit
Objects 

@author: Madi
'''
# ******* Import Modules ******* 
import os
import pandas as pd
import numpy as np
from .quantum_circuit import QuantumCircuit
from .control_pulse import ControlPulse

def write_ctrlp(CntrlPulse, print_ctrlp=True):
	'''
	Writes a .ctrlp file from the ControlPulse
	object passed 
	
	Parameters:
	------------
	CntrlPulse: ControlPulse Object
		Specifies what information to add into
		the .ctrlp file

	Keyword Arguments:
	-------------------
	print_qcirc: Bool
		If True, the function will print out the
		.ctrlp file. If False, nothing will be 
		printed
	
	Returns:
	------------
	Returns a .ctrlp file containing the
	information in the ControlPulse object
	
	Effects:
	------------
	Prints to screen: 
		Prints the .ctrlp file to screen
	
	Requires:
	------------
	* A valid ControlPulse class object as 
	  a parameter defined in the directory
	  qudipy/circuit/control_pulse.py
	'''
	# retrieve all pieces of information
	name = CntrlPulse.name 
	ideal_gate = CntrlPulse.ideal_gate
	p_type = CntrlPulse.pulse_type
	p_length = CntrlPulse.length
	pulse_key = CntrlPulse.ctrl_names
	pulse_vals = list(CntrlPulse.ctrl_pulses.values())
	gates = CntrlPulse.gates
	
	## * Format Control Pulse data into writable strings *
	
	# Format a string of all of the control names
	key_str = ""
	vals_str = ""
	for i, key in enumerate(pulse_key):
		if i == len(pulse_key) -1:
			key_str = key_str + str(key).strip()
		else:
			key_str = key_str + str(key).strip() + "," 
	
	# Format a string of control values
	for j in range(len(pulse_vals[0])):
		vals_str_line = ""
		for i in range(len(pulse_key)):
			if i == (len(pulse_key) - 1):
				vals_str_line = vals_str_line + str(pulse_vals[i][j])
			else:
				vals_str_line = vals_str_line + str(pulse_vals[i][j]) + ", "
		if j == (len(pulse_vals[0]) - 1):
			vals_str = vals_str + vals_str_line
		else:
			vals_str = vals_str + vals_str_line + "\n"
	
	# Check if the ControlPulse contains different simultaneous
	# gates
	if gates != None:
		# Pull out gate sequence
		circ_seq  = gates.circuit_sequence
		
		# Check for Invalid Data
		# Create error message
		err_mess = "QuantumCircuit object stored within " + \
		"a {} ControlPulse object is not a valid simultaneous" +\
		"gate".format(name)
		
		# Check if conditions are correct for a valid 
		# simultaneous gate QuantumCircuit, if not print
		# error message
		if len(circ_seq) > 1:
			raise Exception(err_mess)
			
		if not isinstance(circ_seq[0], tuple):
			raise Exception(err_mess)
		
		# Create string of all gates and their effected qubits
		for gate in circ_seq:
			G = list(map(lambda g: format_single_gate(g), gate))
			# Join them together with a bar | in between gates
			ideal_gate = " | ".join(G)
			
				
	# Write the .ctrlp file
	ctrlp_file = open(name + '.ctrlp', 'w')
	ctrlp_file.writelines(["# {n}.ctrlp\n".format(n = name),
							"Ideal gate: {ig}\n".format(ig = ideal_gate),
							"Pulse type: {pt}\n".format(pt = p_type),
							"Pulse length: {pl} s\n".format(pl = p_length),
							"Control pulses:\n",
							key_str + "\n",
							vals_str])
	
	# Close file to stop making any changes
	ctrlp_file.close()
	
	# Print the file 
	if print_ctrlp:
		directory = os.getcwd() + "/" + name + ".ctrlp"
		f = open(directory, 'r')
		print(f.read())
		f.close()
		
	# return the .ctrlp file
	return(ctrlp_file)


def write_qcirc(QntmCirc, print_qcirc=True):
	'''
	Writes a .qcirc object representing the 
	Quantum Circuit object passed 

	Parameters:
	----------------
	QntmCirc: QuantumCircuit
		The QuantumCircuit object that is to be
		written into a .qcirc file
	
	Keyword Arguments:
	-------------------
	print_qcirc: Bool, optional
		The predicate determining if the user
		wants the .qcirc file to be printed to
		screen. Default is True

	Returns
	----------
	Returns a .qcirc file
	
	Requires:
	----------
	* A valid QuantumCircuit class object is passed
	  as per the class defined with the directory
	  qudipy/circuit/quantum_circuit.py
	'''
	# Pull out all information needed from the circuit
	# and store them as variables.
	name = str(QntmCirc.name)
	n_qubits =str(QntmCirc.n_qubits)
	sequence = QntmCirc.circuit_sequence
	
	# ** Create file using the circuit name **
	qcirc_file = open(name + ".qcirc", 'w')
	
	# *Create string of formated gates*
	string = ""
	for gate in sequence:
		# Consider the instance where there are two different
		# gates acting simultaneously (gates stored in a tuple)
		if isinstance(gate, tuple):
			# Create list of formatted gates
			G = list(map(lambda g: format_single_gate(g), gate))
			# Join them together with a bar | in between gates
			gate_str = " | ".join(G)
		else:
			gate_str = format_single_gate(gate)
		string = string + gate_str + "\n"
	# Remove the extra newline character and add first line 
	# of .qcirc file
	string = string[0:-1]
	# Write in the number of qubits
	write_str = "Number of qubits: {}\n".format(n_qubits) + string
	
	# ** Write the file **
	qcirc_file.write(write_str)
	qcirc_file.close() # Close file
	
	# ** Read and print contents of file **
	directory = os.getcwd() + "/" + name + ".qcirc"
	f = open(directory, 'r')
	print(f.read())
	f.close()
	
	# Return the .qcirc file
	return(qcirc_file)




## ********** HELPER FUNCTIONS ********** 

def format_single_gate(gate):
	'''
	Formats a gate into a writable string
	
	Parameters:
	------------
	gate: Listof(Str, Listof(Int))
		The length 3 list that contains
		one gate inside of a QuantumCircuit 
		sequence
	
	Requires:
	------------
	gate is in the following form:
	['gate_name', 'Ideal Gate', [effected qubits]]
	
	Returns:
	------------
	String
		used to write the gate in the .qcirc file
	'''
	# Retrieve the gate name
	gate_name = gate[1]
	
	# Format effected qubits into a string
	# 1) convert list to list of strings
	eff_qubits_list = list(map(lambda x: str(x), gate[2]))
	# 2) join each elemet with a space between them
	eff_qubits = " ".join(eff_qubits_list)
	
	# Put together the string and return it
	gate_str = gate_name + " " + eff_qubits
	return gate_str

	
	