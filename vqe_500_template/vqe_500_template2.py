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
    
    def variational_ansatz1(params, wires):
        qml.RY(np.pi, wires=1)
        variational_ansatz(params, wires)
        
    def variational_ansatz2(params, wires):
        qml.RY(np.pi, wires=0)
        variational_ansatz(params, wires)
        
    num_qubits = len(H.wires)
    num_param_sets = (2 ** num_qubits) - 1
    
   
    
    dev = qml.device('default.qubit', wires=num_qubits)
    cost_fn0 = qml.ExpvalCost(variational_ansatz, H, dev)
    cost_fn1 = qml.ExpvalCost(variational_ansatz1, H, dev)
    cost_fn2 = qml.ExpvalCost(variational_ansatz2, H, dev)
    
    def ture_cost_fn(params, **kwargs): 
        return cost_fn0(params, **kwargs)+ cost_fn1(params, **kwargs)+ cost_fn2(params, **kwargs)
    
    def cost_fn(params, **kwargs): 
        return cost_fn0(params, **kwargs)+ 0.95*cost_fn1(params, **kwargs)+ 0.9*cost_fn2(params, **kwargs)
    def cost_fn11(params, **kwargs): 
        return cost_fn0(params, **kwargs)+ 0.9*cost_fn1(params, **kwargs)
    
    opt = qml.NesterovMomentumOptimizer(stepsize=0.06)
    # conv_tol = 1e-05
    max_iterations = 500
    
    t = time.time()

    params0 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3))
    for n in range(max_iterations+1):
        params0, prev_w_energy = opt.step_and_cost(cost_fn0, params0)
        w_energy = cost_fn0(params0)
        conv = np.abs(w_energy - prev_w_energy)
        if n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, w_energy))
        if conv <= 1e-07:
            break
        
    params1 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3)) 
    for n in range(max_iterations+1):
        params1, prev_w_energy = opt.step_and_cost(cost_fn11, params1)
        w_energy = cost_fn11(params1)
        conv = np.abs(w_energy - prev_w_energy)
        if n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, w_energy))
        if conv <= 1e-07:
            break
    
    params2 = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=(num_param_sets, 3)) 
    for n in range(max_iterations+1):
        params2, prev_w_energy = opt.step_and_cost(cost_fn, params2)
        w_energy = cost_fn(params2)
        conv = np.abs(w_energy - prev_w_energy)
        if n % 20 == 0:
            print('Iteration = {:},  Energy = {:.8f} Ha'.format(n, w_energy))
        if conv <= 1e-07:
            break
    
    print (time.time() - t)
    e = np.zeros((3,3))
    e[2] = np.sort([cost_fn0(params2),cost_fn1(params2),cost_fn2(params2)])
    e[1] = np.sort([cost_fn0(params1),cost_fn1(params1),cost_fn2(params1)])
    e[0] = np.sort([cost_fn0(params0),cost_fn1(params0),cost_fn2(params0)])
    
    energies = np.min(e, axis=0)
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
