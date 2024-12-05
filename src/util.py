import numpy as np
import numpy.typing as npt 

# Shared typings
type StateVector = npt.NDArray[np.complex128]
type GateMatrix = npt.NDArray[np.complex128]
type GateMatrixArray = npt.NDArray[np.complex128]
type Index = int | slice | list[int]


def convert_index(index: Index) -> list[int]:
    if isinstance(index, int):
        return [index]
    if isinstance(index, slice):
        print(index)
        return list(range(index.start, index.stop, index.step))
    else:
        return index