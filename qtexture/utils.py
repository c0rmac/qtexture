# qtexture/utils.py

import numpy as np


def is_unitary(matrix: np.ndarray, tol: float = 1e-9) -> bool:
    """
    Checks if a matrix is unitary (U * U_dagger = I) within a tolerance.
    This function is used for validating inputs to basis change operations[cite: 139].
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return False
    identity = np.eye(matrix.shape[0])
    # Check both U*U_dagger and U_dagger*U for robustness
    check1 = np.allclose(matrix @ matrix.conj().T, identity, atol=tol)
    check2 = np.allclose(matrix.conj().T @ matrix, identity, atol=tol)
    return check1 and check2


# The following functions demonstrate the interoperability design philosophy.
# They are lightweight converters that avoid a hard dependency on external libraries[cite: 62, 63].

def from_qiskit(dm):
    """
    Utility to convert a Qiskit DensityMatrix object to a qtexture.QuantumState[cite: 253].

    Requires Qiskit to be installed: `pip install qiskit`
    """
    try:
        from qiskit.quantum_info import DensityMatrix
        from .states import QuantumState
    except ImportError:
        raise ImportError("Qiskit is not installed. Please run 'pip install qtexture[qiskit]'.")

    if not isinstance(dm, DensityMatrix):
        raise TypeError("Input must be a qiskit.quantum_info.DensityMatrix object.")

    return QuantumState(dm.data)


def to_qutip(state):
    """
    Utility to convert a qtexture.QuantumState to a QuTiP Qobj[cite: 253].

    Requires QuTiP to be installed: `pip install qutip`
    """
    try:
        from qutip import Qobj
        from .states import QuantumState
    except ImportError:
        raise ImportError("QuTiP is not installed. Please run 'pip install qtexture[qutip]'.")

    if not isinstance(state, QuantumState):
        raise TypeError("Input must be a qtexture.QuantumState object.")

    dims = [[2] * int(np.log2(state.data.shape[0]))] * 2
    return Qobj(state.data, dims=dims)