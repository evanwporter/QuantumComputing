# Page 410
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from QuantumCircuit import QuantumCircuit

# Create a 2-qubit circuit
circuit = QuantumCircuit(2)

# Apply Hadamard gate to both qubits
circuit.h([0, 1])

# Execute the circuit
circuit.execute()

measurement_results = circuit.measure()
print("Measurement results (all qubits):", measurement_results)

measurement_results_q0 = circuit.measure(target_qubits=[0])
print("Measurement results (qubit 0):", measurement_results_q0)

