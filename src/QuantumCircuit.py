import numpy as np
import numpy.typing as npt 
from plot_bloch_sphere import plot_bloch_sphere
from util import *
from typing import Self

class QuantumCircuit:
    num_qubits: int
    state: StateVector
    state_history: list[StateVector]
    gate_queue: GateMatrixArray # 3D Array to hold expanded matrices for each layer

    _gate_queue: GateMatrixArray

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1
        self.state_history = [self.state.copy()]
        self._gate_queue = np.empty((0, 2**num_qubits, 2**num_qubits), dtype=complex) # 3D array
        self.gate_queue = self._gate_queue.copy()

    def _expand_gate(self, gate_matrix: GateMatrix, target_qubits: list[int]) -> GateMatrix:
        full_matrix: GateMatrix = gate_matrix
        for i in range(1, self.num_qubits):
            if i in target_qubits:
                # np.kron is effectively the tensor product
                full_matrix = np.kron(full_matrix, gate_matrix)
            else:
                full_matrix = np.kron(full_matrix, np.eye(2))
        return full_matrix

    def _ensure_layer(self, layer: int):
        current_layers: int = self._gate_queue.shape[0]
        if layer >= current_layers:

            new_layers: int = layer - current_layers + 1
            identity_layer: GateMatrix = np.eye(2**self.num_qubits, dtype=complex)

            # Fill it with identity matrices
            identity_layers: GateMatrixArray = np.repeat(identity_layer[None, :, :], new_layers, axis=0)
            self._gate_queue = np.vstack((self._gate_queue, identity_layers))

    def add_gate(self, gate_matrix: GateMatrix, target_qubits: Index, layer: int):
        normalized_target_qubits: list[int] = convert_index(target_qubits)
        expanded_gate: GateMatrix = self._expand_gate(gate_matrix, normalized_target_qubits)        
        self.add_layer(expanded_gate, layer)
    
    def add_layer(self, gate_matrix: GateMatrix, layer: int):
        if layer == -1: layer = len(self._gate_queue)
        self._ensure_layer(layer)
        self._gate_queue[layer] = gate_matrix

    def execute(self):
        self.gate_queue = self._gate_queue.copy()
        self.state_history = [self.state.copy()] # Flush the history
        for layer in self._gate_queue:
            self.state = layer @ self.state
            self.state_history.append(self.state.copy())

    def h(self, target_qubits: Index, layer: int = -1) -> Self:
        """Haadarmard gate"""
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.add_gate(H, target_qubits, layer)
        return self

    def x(self, target_qubits: Index, layer: int = -1) -> Self:
        """Pauli-X gate"""
        X = np.array([[0, 1], [1, 0]])
        self.add_gate(X, target_qubits, layer=layer)
        return self

    def y(self, target_qubits: Index, layer: int = -1) -> Self:
        """Pauli-Y gate"""
        Y = np.array([[0, -1j], [1j, 0]])
        self.add_gate(Y, target_qubits, layer=layer)
        return self

    def z(self, target_qubits: Index, layer: int = -1) -> Self:
        """Pauli-Z gate"""
        Z = np.array([[1, 0], [0, -1]])
        self.add_gate(Z, target_qubits, layer=layer)
        return self

    def cx(self, control: int, target: int) -> Self:
        """Control Not Gate"""
        size = 2**self.num_qubits
        cx_matrix = np.eye(size, dtype=complex)

        for i in range(size):
            if (i >> control) & 1 == 1:
                target_state = i ^ (1 << target)
                cx_matrix[i, i] = 0
                cx_matrix[i, target_state] = 1

        self.add_layer(cx_matrix, -1)
        return self
    
    def cz(self, control: int, target: int) -> Self:
        """Control Z Gate"""
        size = 2**self.num_qubits
        cz_matrix = np.eye(size, dtype=complex)

        for i in range(size):
            if (i >> control) & 1 == 1 and (i >> target) & 1 == 1:
                cz_matrix[i, i] *= -1

        self.add_layer(cz_matrix, -1)
        return self

    # Bloch Sphere Gates
    def rx(self, theta: float, target_qubits: Index, layer: int = -1) -> Self:
        """Rotation around X-axis"""
        RX = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ])
        self.add_gate(RX, target_qubits, layer)
        return self

    def ry(self, theta: float, target_qubits: Index, layer: int = -1) -> Self:
        """Rotation around Y-axis"""
        RY = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])
        self.add_gate(RY, target_qubits, layer)
        return self

    def rz(self, theta: float, target_qubits: Index, layer: int = -1) -> Self:
        """Rotation around Z-axis"""
        RZ = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ])
        self.add_gate(RZ, target_qubits, layer)
        return self

    def measure_single_qubit(self, state_vector: StateVector, qubit_index: int) -> list[int]:
        probabilities: list[int] = [0, 0]
        for i, amp in enumerate(state_vector):
            if (i >> qubit_index) & 1 == 0:  # Qubit is in |0>
                probabilities[0] += abs(amp)**2
            else:  # Qubit is in |1>
                probabilities[1] += abs(amp)**2
        return probabilities
    
    def measure_combined_probabilities(self, state_vector: StateVector, num_input_qubits: int) -> npt.NDArray[np.float64]:
        num_states: int = 2**num_input_qubits
        probabilities: npt.NDArray[np.float64] = np.zeros(num_states)
        
        index: int
        amplitude: complex
        for index, amplitude in enumerate(state_vector):
            # Bitwise shift removes y qubit and group probabilities
            input_state_index = index >> 1
            probabilities[input_state_index] += abs(amplitude)**2
        
        return probabilities
    
    def measure(self, target_qubits: list[int] | None = None) -> dict[str, float]:
        num_qubits = self.num_qubits
        if target_qubits is None:
            target_qubits = list(range(num_qubits))

        probabilities: npt.NDArray[np.int64] = np.abs(self.state) ** 2
        measurement_results: dict[str, float] = {}

        for index, probability in enumerate(probabilities):
            if probability > 1e-12:  # Ignore negligible probabilities
                # Convert the index to a binary string representing the state
                state_str = format(index, f'0{num_qubits}b')
                # Extract only the bits corresponding to the target qubits
                measured_state: str = ''.join(state_str[i] for i in target_qubits)
                if measured_state in measurement_results:
                    measurement_results[measured_state] += probability
                else:
                    measurement_results[measured_state] = probability

        return measurement_results

    def bloch(self, history: int=-1):
        if self.num_qubits != 1:
            raise KeyError(f"Error num_qubits is greater than one. Block sphere is only able to display a single qubit.")
        plot_bloch_sphere(self.state_history[history])
    
    def _expand_single_qubit_gate(self, gate_matrix: GateMatrix, target_qubit: int) -> GateMatrix:
        """Expands a single-qubit gate to the full state space."""
        full_gate = np.eye(1, dtype=complex)  # Start with a scalar identity
        for i in range(self.num_qubits):
            if i == target_qubit:
                full_gate = np.kron(full_gate, gate_matrix)
            else:
                full_gate = np.kron(full_gate, np.eye(2))
        return full_gate
    

if __name__ == "__main__":
    # Deutsch Problem
    circuit = QuantumCircuit(2)

    circuit.x([1])
    circuit.h([0, 1])
    constantF = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])
    circuit.add_layer(constantF, -1)
    circuit.h([0,1])

    circuit.execute()

    for i, state in enumerate(circuit.state_history):
        print(f"Step {i}: {state}")

    print("Final state vector:", circuit.state)
    print("Measured prob", circuit.measure_single_qubit(circuit.state, 0))
