import numpy as np
from qutip import tensor, sigmax, sigmaz, identity

from typing import List


# Make sure the QuantumState class you provided is importable
from ..states import QuantumState

def get_ising_ground_state(n_spins: int, h: float, j: float = 1.0, tol: float = 1e-9) -> QuantumState:
    """
    Calculates the ground state(s) of the Ising model using QuTiP.

    This version explicitly handles degenerate ground states (e.g., when h=0)
    by returning all states that share the lowest energy.

    Args:
        n_spins: The number of spins in the chain.
        h: The strength of the transverse field.
        j: The coupling strength.
        tol: Numerical tolerance to check for degeneracy.

    Returns:
        A list of QuantumState objects for the ground state(s).
    """
    # Create lists of single-qubit operators
    sx_list = [sigmax()] * n_spins
    sz_list = [sigmaz()] * n_spins
    id_list = [identity(2)] * n_spins

    # Construct the total Hamiltonian
    hamiltonian = 0

    # Add the transverse field term (H_x = -h * sum_i(sx_i))
    for i in range(n_spins):
        op_list = id_list[:]
        op_list[i] = sx_list[i]
        hamiltonian -= h * tensor(op_list)

    # Add the interaction term (H_zz = -j * sum_i(sz_i * sz_{i+1}))
    for i in range(n_spins - 1):
        op_list = id_list[:]
        op_list[i] = sz_list[i]
        op_list[i + 1] = sz_list[i + 1]
        hamiltonian -= j * tensor(op_list)

    # --- CHANGES START HERE ---

    # 1. Use .eigenstates() to get all energies and states, sorted from lowest to highest.
    energies, kets = hamiltonian.eigenstates()

    ground_energy = energies[0]
    ground_kets = []

    # 2. Loop through the results to find all states with the same ground energy.
    for i in range(len(energies)):
        if np.isclose(energies[i], ground_energy, atol=tol):
            ground_kets.append(kets[i])
        else:
            # Since the energies are sorted, we can stop once they no longer match.
            break

    # 3. Convert each ground state ket into a QuantumState density matrix.
    ground_states = []
    for ket in ground_kets:
        density_matrix = ket * ket.dag()
        ground_states.append(QuantumState(density_matrix.full()))

    return ground_states[0]