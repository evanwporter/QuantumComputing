import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pytest
from QuantumCircuit import QuantumCircuit

def test_hadamard_gate():
    circuit = QuantumCircuit(1)

    circuit.h(0)
    circuit.execute()

    expected_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_pauli_x_gate():
    circuit = QuantumCircuit(1)

    circuit.x(0)
    circuit.execute()

    expected_state = np.array([0, 1], dtype=complex)
    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_hadamard_then_x():
    circuit = QuantumCircuit(1)

    circuit.h(0, layer=0)
    circuit.x(0, layer=1)
    circuit.execute()

    expected_state = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)], dtype=complex)
    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_cnot_gate():
    circuit = QuantumCircuit(2)
    circuit.x(0, layer=0)
    circuit.cx(control=0, target=1)
    circuit.execute()

    expected_state = np.array([0, 0, 0, 1], dtype=complex)
    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_cnot_no_action():
    circuit = QuantumCircuit(2)

    circuit.cx(control=0, target=1)
    circuit.execute()

    expected_state = np.array([1, 0, 0, 0], dtype=complex)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_deutsch_constant_function():

    circuit = QuantumCircuit(2)
    circuit.x(1)
    circuit.h([0, 1])
    constant_f = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=complex)
    circuit.add_layer(constant_f, -1)
    circuit.h([0, 1])
    circuit.execute()

    probabilities = circuit.measure_combined_probabilities(circuit.state, 1)

    print(probabilities)
    assert np.isclose(probabilities[0], 0), f"Probabilities mismatch: {probabilities}"

def test_deutsch_balanced_function():
    circuit = QuantumCircuit(2)

    circuit.x(1)

    circuit.h([0, 1])

    balanced_f = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 0, 1],
                           [0, 0, 1, 0]], dtype=complex)
    circuit.add_layer(balanced_f, -1)
    circuit.h([0, 1])
    circuit.execute()
    probabilities = circuit.measure_combined_probabilities(circuit.state, 1)
    assert np.isclose(probabilities[0], 1), f"Probabilities mismatch: {probabilities}"


if __name__ == "__main__":
    pytest.main()
