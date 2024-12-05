import numpy as np
import numpy.typing as npt 

# Shared typings

type StateVector = npt.NDArray[np.complex128]
type GateMatrix = npt.NDArray[np.complex128]
type GateMatrixArray = npt.NDArray[np.complex128]
type Index = int | slice | list[int]