# qtexture/monotones.py

import numpy as np
from .states import QuantumState
from .utils import is_unitary


def _calculate_texture(density_matrix: np.ndarray) -> np.ndarray:
    """
    Private helper function to extract the real part of the density matrix[cite: 113, 114].
    Encapsulates the fundamental operation of the texture theory[cite: 116].
    """
    return np.real(density_matrix)


def calculate_purity_monotone(state: QuantumState) -> float:
    """
    Calculates the texture-based purity monotone[cite: 118, 253].

    This monotone is M_P(ρ) = max(Re(ρ_ij)) - min(Re(ρ_ij))[cite: 30].
    The implementation leverages NumPy's vectorized functions for performance[cite: 132].

    Args:
        state: A qtexture.QuantumState object[cite: 121].

    Returns:
        The scalar value of the purity monotone.
    """
    #if not state.is_valid():
    #    raise ValueError("Input is not a valid quantum state.")

    # Access the internal NumPy array [cite: 123]
    real_part = _calculate_texture(state.data)

    # Use optimized NumPy functions to find min and max [cite: 125]
    max_val = np.max(real_part)
    min_val = np.min(real_part)

    return max_val - min_val  # [cite: 126]


def change_basis(state: QuantumState, unitary: np.ndarray) -> QuantumState:
    """
    Performs a unitary basis transformation on a density matrix[cite: 135, 136].
    The transformation rule is ρ' = UρU†[cite: 137].

    Args:
        state: The initial QuantumState object.
        unitary: A NumPy array representing the unitary transformation U.

    Returns:
        A new QuantumState object in the transformed basis[cite: 141].
    """
    # Validate that the provided matrix is unitary [cite: 139]
    if not is_unitary(unitary):
        raise ValueError("The provided transformation matrix is not unitary.")

    if state.data.shape != unitary.shape:
        raise ValueError("State and unitary matrices have incompatible dimensions.")

    # Core calculation using NumPy's operator for matrix multiplication [cite: 140]
    new_rho_data = unitary @ state.data @ unitary.conj().T

    return QuantumState(new_rho_data)