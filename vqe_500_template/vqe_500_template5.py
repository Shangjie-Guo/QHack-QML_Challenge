#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np
import time

def find_excited_states(H):
    """
    Fill in the missing parts between the # QHACK # markers below. Implement
    a variational method that can find the three lowest energies of the provided
    Hamiltonian.

    Args:
        H (qml.Hamiltonian): The input Hamiltonian

    Returns:
        The lowest three eigenenergies of the Hamiltonian as a comma-separated string,
        sorted from smallest to largest.
    """

    energies = np.zeros(3)

    # QHACK #
    # Use wighted SSVQE:
        
    def variational_ansatz(params, wires):
        qml.templates.subroutines.ArbitraryUnitary(params, wires)
    
    def variational_ansatz1(params, wires):
        qml.RY(np.pi, wires=0)
        variational_ansatz(params, wires)
        
    def variational_ansatz2(params, wires):
        qml.RY(np.pi, wires=0)
        qml.RY(np.pi, wires=1)
        variational_ansatz(params, wires)
        
    num_qubits = len(H.wires)
    num_param_sets = (4 ** (num_qubits)) - 1
    
    
    
    dev = qml.device('default.qubit', wires=num_qubits)
    cost_fn0 = qml.ExpvalCost(variational_ansatz, H, dev)
    cost_fn1 = qml.ExpvalCost(variational_ansatz1, H, dev)
    cost_fn2 = qml.ExpvalCost(variational_ansatz2, H, dev)
    
    def set_w_cost_fn(w):
        def cost_fn(params, **kwargs): 
            return (1+2*w)*cost_fn0(params, **kwargs)+ (1+w)*cost_fn1(params, **kwargs)+cost_fn2(params, **kwargs)
        return cost_fn

    opt = qml.AdamOptimizer(stepsize=0.3)
    conv_tol = 1e-6
    max_iterations = 100
    
    t = time.time()
    params = np.random.uniform(low=-np.pi, high=np.pi, size=(num_param_sets))
    cost_fn_f = set_w_cost_fn(0.25)
    
    for n in range(max_iterations+1):
        params, prev_w_energy = opt.step_and_cost(cost_fn_f, params)
        w_energy = cost_fn_f(params)
        conv = np.abs(w_energy - prev_w_energy)
        if n % 10 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, np.sum(cost_fn0(params)+cost_fn1(params)+cost_fn2(params))))
            print (time.time() - t)
        if conv <= conv_tol:
            break
    
    energies[2] = cost_fn2(params)
    energies[1] = cost_fn1(params)
    energies[0] = cost_fn0(params)
    energies.sort()
    # QHACK #

    return ",".join([str(E) for E in energies])

def pauli_token_to_operator(token):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Helper function to turn strings into qml operators.

    Args:
        token (str): A Pauli operator input in string form.

    Returns:
        A qml.Operator instance of the Pauli.
    """
    qubit_terms = []

    for term in token:
        # Special case of identity
        if term == "I":
            qubit_terms.append(qml.Identity(0))
        else:
            pauli, qubit_idx = term[0], term[1:]
            if pauli == "X":
                qubit_terms.append(qml.PauliX(int(qubit_idx)))
            elif pauli == "Y":
                qubit_terms.append(qml.PauliY(int(qubit_idx)))
            elif pauli == "Z":
                qubit_terms.append(qml.PauliZ(int(qubit_idx)))
            else:
                print("Invalid input.")

    full_term = qubit_terms[0]
    for term in qubit_terms[1:]:
        full_term = full_term @ term

    return full_term


def parse_hamiltonian_input(input_data):
    """
    DO NOT MODIFY anything in this function! It is used to judge your solution.

    Turns the contents of the input file into a Hamiltonian.

    Args:
        filename(str): Name of the input file that contains the Hamiltonian.

    Returns:
        qml.Hamiltonian object of the Hamiltonian specified in the file.
    """
    # Get the input
    coeffs = []
    pauli_terms = []

    # Go through line by line and build up the Hamiltonian
    for line in input_data.split("S"):
        line = line.strip()
        tokens = line.split(" ")

        # Parse coefficients
        sign, value = tokens[0], tokens[1]

        coeff = float(value)
        if sign == "-":
            coeff *= -1
        coeffs.append(coeff)

        # Parse Pauli component
        pauli = tokens[2:]
        pauli_terms.append(pauli_token_to_operator(pauli))

    return qml.Hamiltonian(coeffs, pauli_terms)


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Turn input to Hamiltonian
    H = parse_hamiltonian_input(sys.stdin.read())

    # Send Hamiltonian through VQE routine and output the solution
    lowest_three_energies = find_excited_states(H)
    print(lowest_three_energies)
