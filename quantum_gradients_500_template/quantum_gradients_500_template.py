#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    @qml.qnode(dev)
    def state(params):
        variational_circuit(params)
        return qml.state()
    
    state1 = state(params).conj()
    def overlap(params_s):
        # overlap
        return np.abs(np.dot(state1,state(params_s)))**2
    
    s = np.pi/2
    def shift(params,i,j):
        p = np.zeros((4,6))
        p[0] = params
        p[1] = params
        p[2] = params
        p[3] = params
        p[0,i] += s
        p[0,j] += s
        p[1,i] += s
        p[1,j] -= s
        p[2,i] -= s
        p[2,j] += s
        p[3,i] -= s
        p[3,j] -= s
        return p
        
    F = np.zeros((6,6))
    
    for i in range(6):
        for j in range(6):
            params_s = shift(params,i,j)
            F[i,j] = (-overlap(params_s[0])+overlap(params_s[1])+overlap(params_s[2])-overlap(params_s[3])) / 8

    gradient = np.zeros(6)
    denom = 2 * np.sin(s)
    for i in range(6):
        se = np.zeros_like(params)
        se[i] = s
        gradient[i] = (float(qnode(params+se)) - float(qnode(params-se))) /denom
    
    natural_grad = np.matmul(np.linalg.inv(F), gradient)
    
    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")
