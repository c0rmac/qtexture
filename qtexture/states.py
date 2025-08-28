# qtexture/states.py

import numpy as np
from typing import Union, List


class QuantumState:
    """
    The core class for representing a quantum state's density matrix[cite: 253].
    This class serves as the central data structure for the library[cite: 74].
    """

    def __init__(self, data: Union[np.ndarray, List[List[complex]]]):
        """
        Initializes the QuantumState object.

        The constructor is flexible, accepting various data formats and
        converting them to the canonical internal representation[cite: 78, 79].

        Args:
            data: A representation of the density matrix, e.g., a NumPy array
                  or a list of lists.
        """
        # Internal representation is a high-precision complex NumPy array [cite: 77]
        self.data = np.asarray(data)

        # If the input was not complex, default to complex128 for general use.
        if self.data.dtype not in [np.complex64, np.complex128]:
            self.data = self.data.astype(np.complex128)

        # Initial validation to ensure the data has a valid matrix shape [cite: 80]
        if self.data.ndim != 2 or self.data.shape[0] != self.data.shape[1]:
            raise ValueError("Input data must be a square 2D matrix.")

        dim = self.data.shape[0]
        if not (dim > 0 and (dim & (dim - 1)) == 0):
            # This check is useful for qubit systems but not strictly required
            # by the theory. It's a good practice for usability.
            import warnings
            warnings.warn("Matrix dimension is not a power of 2, which is unusual for qubit systems.")

    @property
    def is_hermitian(self) -> bool:
        """
        Checks if the density matrix is Hermitian (ρ = ρ†)[cite: 83].
        Uses np.allclose for robust floating-point comparison[cite: 91].
        """
        return np.allclose(self.data, self.data.conj().T)

    @property
    def trace(self) -> complex:
        """Returns the trace of the density matrix using np.trace()[cite: 84]."""
        return np.trace(self.data)

    @property
    def is_positive_semidefinite(self, tol: float = 1e-9) -> bool:
        """
        Checks if the matrix is positive semi-definite using a fast
        Cholesky decomposition with an eigenvalue fallback for robustness.

        Args:
            tol: The numerical tolerance for the eigenvalue check.
        """
        # Cholesky decomposition requires a Hermitian matrix.
        if not self.is_hermitian:
            return False

        try:
            # np.linalg.cholesky is much faster than eigvalsh. It will succeed
            # if the matrix is positive definite.
            np.linalg.cholesky(self.data)
            return True
        except np.linalg.LinAlgError:
            # Cholesky fails for matrices that are not positive definite.
            # This can happen if the state is positive SEMI-definite (has
            # zero eigenvalues, like a pure state) or if it's invalid.
            # We fall back to the slower but more general eigenvalue check
            # only in this case.
            eigenvalues = np.linalg.eigvalsh(self.data)
            return np.all(eigenvalues >= -tol)

    @property
    def purity(self) -> float:
        """Calculates the purity of the state, Tr(ρ^2)[cite: 86]."""
        return np.real(np.trace(self.data @ self.data))

    def is_valid(self, tol: float = 1e-9) -> bool:
        """
        Performs a comprehensive check for physical validity[cite: 87, 88].
        A valid density matrix must be Hermitian, have a trace of 1, and be
        positive semi-definite[cite: 89].

        Args:
            tol: The numerical tolerance for floating-point comparisons[cite: 90].

        Returns:
            True if the state is a valid density matrix, False otherwise.
        """
        # Use np.isclose for robust scalar comparison [cite: 91]
        trace_is_one = np.isclose(self.trace, 1.0, atol=tol)
        return self.is_hermitian and trace_is_one and self.is_positive_semidefinite


def create_ghz_state(num_qubits: int) -> QuantumState:
    """
    Generates the Greenberger–Horne–Zeilinger (GHZ) state[cite: 96].
    |GHZ⟩ = (1/√2) * (|0...0⟩ + |1...1⟩)
    """
    if num_qubits < 2:
        raise ValueError("GHZ state requires at least 2 qubits.")
    dim = 2 ** num_qubits
    psi = np.zeros(dim)
    psi[0] = 1 / np.sqrt(2)
    psi[-1] = 1 / np.sqrt(2)
    rho = np.outer(psi, psi.conj())
    return QuantumState(rho)


def create_w_state(num_qubits: int) -> QuantumState:
    """
    Generates the W state[cite: 97].
    |W⟩ = (1/√N) * (|10...0⟩ + |01...0⟩ + ... + |0...01⟩)
    """
    if num_qubits < 2:
        raise ValueError("W state requires at least 2 qubits.")
    dim = 2 ** num_qubits
    psi = np.zeros(dim)
    for i in range(num_qubits):
        psi[2 ** i] = 1 / np.sqrt(num_qubits)
    rho = np.outer(psi, psi.conj())
    return QuantumState(rho)


def create_maximally_mixed_state(num_qubits: int) -> QuantumState:
    """
    Generates the maximally mixed state (ρ = I / 2^N)[cite: 98].
    This state should have zero texture and serves as a key test case[cite: 98].
    """
    if num_qubits < 1:
        raise ValueError("Must have at least 1 qubit.")
    dim = 2 ** num_qubits
    rho = np.eye(dim) / dim
    return QuantumState(rho)