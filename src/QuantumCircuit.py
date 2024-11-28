import numpy as np
import numpy.typing as npt 

type StateVector = npt.NDArray[np.complex128]
type GateMatrix = npt.NDArray[np.complex128] # Scalar if Num_Qubits is 0 
type GateMatrixArray = npt.NDArray[np.complex128]
type Index = int | slice | list[int]


class QuantumCircuit:
    num_qubits: int
    state: StateVector
    state_history: list[StateVector]
    gate_queue: GateMatrixArray # 3D Array to hold expanded matrices for each layer

    def __init__(self, num_qubits: int):
        
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1
        self.state_history = [self.state.copy()]
        self.gate_queue = np.empty((0, 2**num_qubits, 2**num_qubits), dtype=complex) # 3D array

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
        current_layers: int = self.gate_queue.shape[0]
        if layer >= current_layers:

            new_layers: int = layer - current_layers + 1
            identity_layer: GateMatrix = np.eye(2**self.num_qubits, dtype=complex)

            # Fill it with identity matrices
            identity_layers: GateMatrixArray = np.repeat(identity_layer[None, :, :], new_layers, axis=0)
            self.gate_queue = np.vstack((self.gate_queue, identity_layers))

    def add_gate(self, gate_matrix: GateMatrix, target_qubits: Index, layer: int):
        normalized_target_qubits: list[int] = self._convert_index(target_qubits)
        expanded_gate: GateMatrix = self._expand_gate(gate_matrix, normalized_target_qubits)        
        self.add_layer(expanded_gate, layer)
    
    def add_layer(self, gate_matrix: GateMatrix, layer: int):
        if layer == -1: layer = len(self.gate_queue)
        self._ensure_layer(layer)
        self.gate_queue[layer] = gate_matrix

    def _convert_index(self, index: Index) -> list[int]:
        if isinstance(index, int):
            return [index]
        if isinstance(index, slice):
            return list(range(index.start, index.stop, index.step))
        else:
            return index

    def execute(self):
        for layer in self.gate_queue:
            self.state = layer @ self.state
            self.state_history.append(self.state.copy())

    def h(self, target_qubits: Index, layer: int = -1):
        """Haadarmard gate"""
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.add_gate(H, target_qubits, layer)

    def x(self, target_qubits: Index, layer: int = -1):
        """Pauli-X gate"""
        X = np.array([[0, 1], [1, 0]])
        self.add_gate(X, target_qubits, layer=layer)

    def cx(self, control: int, target: int):
        """Control Not Gate"""
        # Unsure if this works!!!

        if control == target:
            raise ValueError("Control and target qubits must be different.")
        
        size = 2**self.num_qubits
        cnot_matrix = np.eye(size, dtype=complex)
        
        for i in range(size):
            binary = format(i, f'0{self.num_qubits}b')  # Binary representation
            if binary[control] == '1':  # Control qubit is |1⟩
                # Flip the target qubit
                target_index = int(binary[:target] + str(1 - int(binary[target])) + binary[target+1:], 2)
                cnot_matrix[i, i] = 0
                cnot_matrix[i, target_index] = 1
        
        self.add_layer(cnot_matrix, -1)

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
