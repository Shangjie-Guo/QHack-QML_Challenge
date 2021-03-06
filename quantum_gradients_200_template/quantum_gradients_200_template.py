#! /usr/bin/python3

import sys
import pennylane as qml
import numpy as np


def gradient_200(weights, dev):
    r"""This function must compute the gradient *and* the Hessian of the variational
    circuit using the parameter-shift rule, using exactly 51 device executions.
    The code you write for this challenge should be completely contained within
    this function between the # QHACK # comment markers.

    Args:
        weights (array): An array of floating-point numbers with size (5,).
        dev (Device): a PennyLane device for quantum circuit execution.

    Returns:
        tuple[array, array]: This function returns a tuple (gradient, hessian).

            * gradient is a real NumPy array of size (5,).

            * hessian is a real NumPy array of size (5, 5).
    """

    @qml.qnode(dev, interface=None)
    def circuit(w):
        for i in range(3):
            qml.RX(w[i], wires=i)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RY(w[3], wires=1)

        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 0])

        qml.RX(w[4], wires=2)

        return qml.expval(qml.PauliZ(0) @ qml.PauliZ(2))

    gradient = np.zeros([5], dtype=np.float64)
    hessian = np.zeros([5, 5], dtype=np.float64)

    # QHACK #
    grad_pm = np.zeros([5, 2], dtype=np.float64)
    s = 2*np.pi/3
    denom = 2 * np.sin(s)
    cen = float(circuit(weights))
    for i in range(5):
        se = np.zeros_like(weights)
        se[i] = s
        grad_pm[i,0] = float(circuit(weights+se))
        grad_pm[i,1] = float(circuit(weights-se))
        gradient[i] = (grad_pm[i,0]  - grad_pm[i,1]) / denom
        hessian[i,i] = (grad_pm[i,0] + grad_pm[i,1] - 2*cen) / (denom**2)
    
    for i in range(4):
        for j in range(i+1):
            ses = np.zeros_like(weights)
            ses[i+1] = s
            ses[j] = s
            sea = np.zeros_like(weights)
            sea[i+1] = s
            sea[j] = -s
            hessian[i+1,j] = (float(circuit(weights+ses))+float(circuit(weights-ses))-float(circuit(weights+sea))-float(circuit(weights-sea))) / (denom**2)
            hessian[j,i+1] = hessian[i+1,j]
    # QHACK #

    return gradient, hessian, circuit.diff_options["method"]


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    weights = sys.stdin.read()
    weights = weights.split(",")
    weights = np.array(weights, float)

    dev = qml.device("default.qubit", wires=3)
    gradient, hessian, diff_method = gradient_200(weights, dev)

    print(
        *np.round(gradient, 10),
        *np.round(hessian.flatten(), 10),
        dev.num_executions,
        diff_method,
        sep=","
    )
