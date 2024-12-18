import numpy as np
import numpy.typing as npt 

# Shared typings
type StateVector = npt.NDArray[np.complex128]
type GateMatrix = npt.NDArray[np.complex128]
type GateMatrixArray = npt.NDArray[np.complex128]
type index_t = int | slice | list[int]

def convert_index(index: index_t) -> list[int]:
    if isinstance(index, int):
        return [index]
    if isinstance(index, slice):
        print(index)
        return list(range(index.start, index.stop, index.step))
    else:
        return index
    
def generate_states(n: int) -> list[str]:
    """
    Lists out every state. 

    Example: For two qubits it would return

    [00, 01, 10, 11]
    """
    return [f"{i:0{n}b}" for i in range(2**n)]