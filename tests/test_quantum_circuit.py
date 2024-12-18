# type: ignore
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import numpy as np
import pytest
from QuantumCircuit import QuantumCircuit

from qiskit import QuantumCircuit as QK
from qiskit import transpile
from qiskit_aer import AerSimulator

def run_qiskit_circuit(qiskit_circuit):
    simulator = AerSimulator()
    qiskit_circuit.save_statevector()
    compiled_circuit = transpile(qiskit_circuit, simulator)
    result = simulator.run(compiled_circuit).result()
    return result.data()["statevector"]

def test_hadamard_gate():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.execute()

    qc = QK(1)
    qc.h(0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_pauli_x_gate():
    circuit = QuantumCircuit(1)
    circuit.x(0)
    circuit.execute()

    qc = QK(1)
    qc.x(0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_hadamard_then_x():
    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.x(0)
    circuit.execute()

    qc = QK(1)
    qc.h(0)
    qc.x(0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_pauli_y_gate():
    circuit = QuantumCircuit(1)
    circuit.y(0)
    circuit.execute()

    qc = QK(1)
    qc.y(0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_pauli_z_gate():
    circuit = QuantumCircuit(1)
    circuit.z(0)
    circuit.execute()

    qc = QK(1)
    qc.z(0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_rotation_x_gate():
    circuit = QuantumCircuit(1)
    circuit.rx(np.pi, 0)
    circuit.execute()

    qc = QK(1)
    qc.rx(np.pi, 0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_rotation_y_gate():
    circuit = QuantumCircuit(1)
    circuit.ry(np.pi / 2, 0)
    circuit.execute()

    qc = QK(1)
    qc.ry(np.pi / 2, 0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_rotation_z_gate():
    circuit = QuantumCircuit(1)
    circuit.rz(np.pi, 0)
    circuit.execute()

    qc = QK(1)
    qc.rz(np.pi, 0)
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

def test_cnot_gate():
    circuit = QuantumCircuit(2)
    circuit.x(0)
    circuit.cx(control=0, target=1)
    circuit.execute()

    qc = QK(2)
    qc.x(0)
    qc.cx(0, 1)
    expected_state = run_qiskit_circuit(qc)

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

    qc = QK(2)
    qc.x(1)
    qc.h([0, 1])
    qc.unitary(constant_f, [0, 1])
    qc.h([0, 1])
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

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

    qc = QK(2)
    qc.x(1)
    qc.h([0, 1])
    qc.unitary(balanced_f, [0, 1])
    qc.h([0, 1])
    expected_state = run_qiskit_circuit(qc)

    assert np.allclose(circuit.state, expected_state), f"State mismatch: {circuit.state}"

if __name__ == "__main__":
    pytest.main()
