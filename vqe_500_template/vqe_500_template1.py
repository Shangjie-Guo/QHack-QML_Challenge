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
    # Use Q-deflation:
        
    def variational_ansatz(params, wires):
        n_qubits = len(wires)
        n_rotations = len(params)
        
        if n_rotations > 1:
            n_layers = n_rotations // n_qubits
            n_extra_rots = n_rotations - n_layers * n_qubits
            
            # Alternating layers of unitary rotations on every qubit followed by a
            # ring cascade of CNOTs.
            for layer_idx in range(n_layers):
                layer_params = params[layer_idx * n_qubits : layer_idx * n_qubits + n_qubits, :]
                qml.broadcast(qml.Rot, wires, pattern="single", parameters=layer_params)
                qml.broadcast(qml.CNOT, wires, pattern="ring")
                
                # There may be "extra" parameter sets required for which it's not necessarily
                # to perform another full alternating cycle. Apply these to the qubits as needed.
                extra_params = params[-n_extra_rots:, :]
                extra_wires = wires[: n_qubits - 1 - n_extra_rots : -1]
                qml.broadcast(qml.Rot, extra_wires, pattern="single", parameters=extra_params)
        else:
            # For 1-qubit case, just a single rotation to the qubit
            qml.Rot(*params[0], wires=wires[0])
        
    num_qubits = len(H.wires)
    num_param_sets = (2 ** num_qubits) - 1
    params = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(3, num_param_sets, 3))
    
    dev = qml.device('default.qubit', wires=num_qubits)
    cost_fn0 = qml.ExpvalCost(variational_ansatz, H, dev)

    
    @qml.qnode(dev)
    def circuit(params, wires):
        variational_ansatz(params, wires)
        return qml.state()
    
    def cost_fn(params, **kwargs): 
        s=[]
        for i in range(3):
            s.append(circuit(params[i], wires=list(range(num_qubits))))
        o=[]
        for i in range(3):
            o.append(np.abs(np.dot(np.real(s[i])-np.imag(s[i]), s[(i+1)%3]))**2)
        return cost_fn0(params[0], **kwargs) + cost_fn0(params[1], **kwargs)+ cost_fn0(params[2], **kwargs) + 100*np.sum(o)
    opt = qml.GradientDescentOptimizer(stepsize=0.05)
    conv_tol = 1e-06
    max_iterations = 1000
    
    t = time.time()

    
    for n in range(max_iterations):
        params, prev_w_energy = opt.step_and_cost(cost_fn, params)
        w_energy = cost_fn(params)
        conv = np.abs(w_energy - prev_w_energy)
        if n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, w_energy))
        if conv <= conv_tol:
            break
    print (time.time() - t)

    for i in range(3):
        energies[i] = cost_fn0(params[i])

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
