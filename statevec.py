import numpy as np


class StateVector:
    def __init__(self, n_qubits) -> None:
        self.n_qubits = n_qubits
        self.state = None

        self.set_state()
        self.shape = self.state.shape
    
    def set_state(self, state=None, random=False) -> None:
        if random:
            state = np.random.randn(2 ** self.n_qubits) + 1j * np.random.randn(2 ** self.n_qubits)
            state = state / np.linalg.norm(state, ord=2)
        elif state is None:
            state = np.zeros(2 ** self.n_qubits, dtype=np.cfloat)
            state[0] = 1
        self.state = state

    def __repr__(self) -> str:
        return f"StateVector({self.state})"

    def apply_1q_gate(self, gate, wire) -> None:
        alphabet = "abcdefghijklmnopqerstuvwxyz"
        gate = gate.reshape([2] * 2)
        current_state = self.state.reshape([2] * self.n_qubits)

        init_indices = alphabet[:self.n_qubits].replace(alphabet[wire], "Y")
        final_indices = alphabet[:self.n_qubits].replace(alphabet[wire], "X")
        einsum_str = f"XY,{init_indices} -> {final_indices}"
        new_state = np.einsum(einsum_str, gate, current_state )

        self.state = new_state.reshape(-1)
    
    def apply_2q_gate(self, gate, wires) -> None:
        alphabet = "abcdefghijklmnopqerstuvwxyz"
        gate = gate.reshape([2] * 4)
        current_state = self.state.reshape([2] * self.n_qubits)

        init_indices = alphabet[:self.n_qubits].replace(
            alphabet[wires[0]], "Y").replace(alphabet[wires[1]], "Z")
        final_indices = alphabet[:self.n_qubits].replace(
            alphabet[wires[0]], "W").replace(alphabet[wires[1]], "X")
        einsum_str = f"WXYZ,{init_indices} -> {final_indices}"

        new_state = np.einsum(einsum_str, gate, current_state )
        # new_state = np.tensordot(gate, current_state, axes=([-2,-1],wires))
        
        self.state = new_state.reshape(-1)

    def density_matrix(self) -> np.ndarray:
        return np.outer(self.state, self.state)
    
    def probs(self) -> np.ndarray:
        return np.abs(self.state).real ** 2

    def measure(self, observable=None) -> np.ndarray:
        if observable is None:
            return self.probs()
        else:
            #Not Implemented Yet
            pass
    
    def CNOT(self, wires=None) -> None:
        if wires is None:
            for i in range(self.n_qubits):
                self.CNOT([i,(i + 1) % self.n_qubits])
        elif self.n_qubits > 1:
            cnot = np.zeros((4,4), , dtype=np.cfloat)
            cnot[0,0] = cnot[1,1] = cnot[2,3] = cnot[3,2] = 1
            self.apply_2q_gate(cnot,wires)
    
    def Hadamard(self, wires=None) -> None:
        if wires is None:
            wires = range(self.n_qubits)

        if hasattr(wires, "__iter__"):
            for i in wires:
                self.Hadamard(i)
        else:
            h = np.array([[1,1],[1,-1]], dtype=np.cfloat) / np.sqrt(2)
            self.apply_1q_gate(h,wires)

    def RZ(self, theta, wires=None) -> None:
        if wires is None:
            wires = range(self.n_qubits)

        if hasattr(wires, "__iter__"):                   
            for i in wires:
                angle = theta[i] if hasattr(theta, "__iter__") else theta
                self.RZ(angle, i)

        else:
            e = np.exp(-1j * theta / 2)
            rz = np.array([[e,0],[0,e.conj()]], dtype=np.cfloat)
            self.apply_1q_gate(rz,wires)

    def RX(self, theta, wires=None) -> None:
        self.Hadamard(wires=wires)
        self.RZ(theta, wires)
        self.Hadamard(wires=wires)

    def RY(self, theta, wires=None) -> None:
        self.RZ(np.pi/2, wires=wires)
        self.RX(theta, wires)
        self.RZ(-np.pi/2, wires=wires)