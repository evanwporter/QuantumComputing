import numpy as np
import numpy.typing as npt
from plotting import plot_bloch_sphere, plot_measurement_results
import util
from util import StateVector, GateMatrix, GateMatrixArray, index_t
from typing import Self

class QuantumCircuit:
    num_qubits: int
    state: StateVector
    state_history: list[StateVector]
    gate_queue: GateMatrixArray # 3D Array to hold expanded matrices for each layer
    _gate_queue: GateMatrixArray
    _executed: bool = False

    def __init__(self, num_qubits: int):
        self.num_qubits = num_qubits
        self.state = np.zeros(2**num_qubits, dtype=complex)
        self.state[0] = 1
        self.state_history = [self.state.copy()]
        self._gate_queue = np.empty((0, 2**num_qubits, 2**num_qubits), dtype=complex) # 3D array
        self.gate_queue = self._gate_queue.copy()

    def _expand_gate(self, gate_matrix: GateMatrix, target_qubits: list[int]) -> GateMatrix:
        full_gate: GateMatrix = np.eye(1, dtype=complex)
        
        for i in range(self.num_qubits):
            if i in target_qubits:
                full_gate = np.kron(gate_matrix, full_gate)
            else:
                full_gate = np.kron(np.eye(2, dtype=complex), full_gate)
        
        return full_gate
    
    def _expand_partial_gate(self, gate_matrix: GateMatrix, target_qubits: list[int]) -> GateMatrix:
        num_target_qubits = int(np.log2(gate_matrix.shape[0]))

        if len(target_qubits) != num_target_qubits:
            raise ValueError(f"Gate acts on {num_target_qubits} qubits, but {len(target_qubits)} were provided.")

        full_gate = np.eye(1, dtype=complex)

        for i in range(self.num_qubits):
            if i in target_qubits:
                full_gate = np.kron(full_gate, gate_matrix)
            else:
                full_gate = np.kron(full_gate, np.eye(2, dtype=complex))

        return full_gate


    def _ensure_layer(self, layer: int):
        current_layers: int = self._gate_queue.shape[0]
        if layer >= current_layers:

            new_layers: int = layer - current_layers + 1
            identity_layer: GateMatrix = np.eye(2**self.num_qubits, dtype=complex)

            # Fill it with identity matrices
            identity_layers: GateMatrixArray = np.repeat(identity_layer[None, :, :], new_layers, axis=0)
            self._gate_queue = np.vstack((self._gate_queue, identity_layers))

    def add_gate(self, gate_matrix: GateMatrix, target_qubits: index_t, layer: int):
        normalized_target_qubits: list[int] = util.convert_index(target_qubits)
        expanded_gate: GateMatrix = self._expand_gate(gate_matrix, normalized_target_qubits)
        self.add_layer(expanded_gate, layer)

    def add_layer(self, gate_matrix: GateMatrix, layer: int):
        if layer == -1: layer = len(self._gate_queue)
        self._ensure_layer(layer)
        self._gate_queue[layer] = gate_matrix

    def execute(self):
        self._executed = True
        self.clear()
        for layer in self._gate_queue:
            self.state = layer @ self.state
            self.state_history.append(self.state.copy())

    def clear(self):
        self.gate_queue = self._gate_queue.copy()
        self.state_history = [self.state.copy()] # Flush the history

    def h(self, target_qubits: index_t, layer: int = -1) -> Self:
        """Haadarmard gate"""
        H = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
        self.add_gate(H, target_qubits, layer)
        return self

    def x(self, target_qubits: index_t, layer: int = -1) -> Self:
        """Pauli-X gate"""
        X = np.array([[0, 1], [1, 0]])
        self.add_gate(X, target_qubits, layer=layer)
        return self

    def y(self, target_qubits: index_t, layer: int = -1) -> Self:
        """Pauli-Y gate"""
        Y = np.array([[0, -1j], [1j, 0]])
        self.add_gate(Y, target_qubits, layer=layer)
        return self

    def z(self, target_qubits: index_t, layer: int = -1) -> Self:
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
        """
        Control Z Gate
        
        Applies Z gate to target when control is 1.
        """
        size = 2**self.num_qubits
        cz_matrix = np.eye(size, dtype=complex)

        for i in range(size):
            if (i >> control) & 1 == 1 and (i >> target) & 1 == 1:
                cz_matrix[i, i] *= -1

        self.add_layer(cz_matrix, -1)
        return self

    # Bloch Sphere Gates
    def rx(self, theta: float, target_qubits: index_t, layer: int = -1) -> Self:
        """Rotation around X-axis"""
        RX = np.array([
            [np.cos(theta / 2), -1j * np.sin(theta / 2)],
            [-1j * np.sin(theta / 2), np.cos(theta / 2)]
        ])
        self.add_gate(RX, target_qubits, layer)
        return self

    def ry(self, theta: float, target_qubits: index_t, layer: int = -1) -> Self:
        """Rotation around Y-axis"""
        RY = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ])
        self.add_gate(RY, target_qubits, layer)
        return self

    def rz(self, theta: float, target_qubits: index_t, layer: int = -1) -> Self:
        """Rotation around Z-axis"""
        RZ = np.array([
            [np.exp(-1j * theta / 2), 0],
            [0, np.exp(1j * theta / 2)]
        ])
        self.add_gate(RZ, target_qubits, layer)
        return self
    
    def to_gate(self) -> GateMatrix:
        """Combine all gates in the circuit into a single gate."""

        unitary = np.eye(2**self.num_qubits, dtype=complex)
        
        for layer in self._gate_queue:
            unitary = layer @ unitary
        
        self.clear()
        return unitary
    
    def bloch(self, history: int=-1):
        if self.num_qubits != 1:
            raise KeyError(f"Error num_qubits is greater than one. Block sphere is only able to display a single qubit.")
        plot_bloch_sphere(self.state_history[history])
    
    @property
    def probabilities(self) -> npt.NDArray[np.int64]:
        return np.real( # force conversion into a real number
            np.square(np.abs(self.state))
         )

    def toffoli(self) -> Self:
        """
        Implement a 2-control Toffoli gate.
        """
        # https://www.cs.sfu.ca/~meamy/Teaching/f22/cmpt981/Lecture%205.pdf
        toff: GateMatrix = np.eye(8, dtype=complex)

        toff[[6, 7], [6, 7]] = 0
        toff[[6, 7], [7, 6]] = 1

        self.add_layer(toff, -1)

        return self        

        
    def measure(self, target_qubits: index_t | None = None, num_shots: int = 1024) -> dict[str, int]:
        if not self._executed: self.execute()  # Get the latest state

        if target_qubits is None:
            target_qubits = list(range(self.num_qubits))
        else:
            target_qubits = util.convert_index(target_qubits)

        # Generate all the possible states ie: |00>, |01>, etc...
        basis_states = util.generate_states(self.num_qubits)

        # If measuring a subset of total number bits then sum up the non measured bits.
        # ie: if we have two qubits and we were trying to measure the prob of the first qubit
        #     we could sum up |00> and |01>, and bit the result under |0>
        marginal_basis = [
            ''.join(state[q] for q in target_qubits) for state in basis_states
        ]

        marginal_probs: dict[str, float] = {}
        for full_state, marginal_state in zip(basis_states, marginal_basis):
            marginal_probs[marginal_state] = marginal_probs.get(marginal_state, 0) + self.probabilities[int(full_state, 2)]

        total_prob = sum(marginal_probs.values())
        marginal_probs = {state: prob / total_prob for state, prob in marginal_probs.items()}

        # Simulate measurement outcomes
        outcomes = np.random.choice(
            list(marginal_probs.keys()),
            size=num_shots,
            p=list(marginal_probs.values())
        )

        # Count occurrences of each outcome
        measurement_results = {state: 0 for state in marginal_probs}
        for outcome in outcomes:
            measurement_results[outcome] += 1

        return measurement_results


    def bar_chart(self, target_qubits: index_t | None):
        measurement = {k: v for k, v in self.measure(target_qubits).items() if v != 0}
        plot_measurement_results(measurement)
