# qtexture/validation/ising_model.py

import numpy as np
from scipy.sparse.linalg import eigsh
from typing import List
from ..states import QuantumState


def get_ising_mpo(n_spins: int, h: float, j: float = 1.0) -> List[np.ndarray]:
    """
    Constructs the Matrix Product Operator (MPO) for the 1D Transverse
    Field Ising Model Hamiltonian.
    """
    sx = np.array([[0, 1], [1, 0]])
    sz = np.array([[1, 0], [0, -1]])
    identity = np.eye(2)

    w = np.zeros((3, 3, 2, 2))
    w[0, 0] = w[2, 2] = identity
    w[0, 1] = -j * sz
    w[0, 2] = -h * sx
    w[1, 2] = sz

    mpo = [w] * n_spins
    mpo[0] = w[0:1, :, :, :]
    mpo[-1] = w[:, 2:3, :, :]

    return mpo


# In qtexture/validation/ising_model.py, replace the entire _DMRG class

class _DMRG:
    """A simplified DMRG implementation. Kept as a private class."""

    def __init__(self, n_spins: int, bond_dimension: int, h: float, j: float = 1.0):
        self.n_spins = n_spins
        self.bond_dimension = bond_dimension
        self.mpo = get_ising_mpo(n_spins, h, j)
        self.mps = self._initialize_mps()

    def _initialize_mps(self) -> List[np.ndarray]:
        mps = []
        mps.append(np.random.rand(1, 2, min(self.bond_dimension, 2)))
        for i in range(1, self.n_spins - 1):
            dim_in = mps[-1].shape[2]
            dim_out = min(self.bond_dimension, dim_in * 2)
            mps.append(np.random.rand(dim_in, 2, dim_out))
        mps.append(np.random.rand(mps[-1].shape[2], 2, 1))
        return mps

    def run(self, num_sweeps: int, tol: float = 1e-8):
        right_envs = [np.ones((1, 1, 1))] * self.n_spins
        for i in range(self.n_spins - 1, 0, -1):
            right_envs[i - 1] = self._contract_env(self.mps[i], self.mpo[i], right_envs[i], 'right')

        left_env = np.ones((1, 1, 1))
        last_energy = None
        for sweep in range(num_sweeps):
            for i in range(self.n_spins - 1):
                energy, new_A, new_B = self._optimize_two_sites(left_env, right_envs[i + 1], i)
                self.mps[i], self.mps[i + 1] = new_A, new_B
                left_env = self._contract_env(new_A, self.mpo[i], left_env, 'left')

            for i in range(self.n_spins - 1, 0, -1):
                energy, new_A, new_B = self._optimize_two_sites(left_env, right_envs[i], i - 1)
                self.mps[i - 1], self.mps[i] = new_A, new_B
                right_envs[i - 1] = self._contract_env(new_B, self.mpo[i], right_envs[i], 'right')

            print(f"Sweep {sweep + 1}/{num_sweeps}, Energy = {energy:.8f}")
            if last_energy and abs(energy - last_energy) < tol:
                print("Convergence reached.")
                break
            last_energy = energy

    def _optimize_two_sites(self, left_env, right_env, site_idx):
        """Performs the core optimization step on two adjacent sites."""
        w1, w2 = self.mpo[site_idx], self.mpo[site_idx + 1]

        H_eff_flat = np.einsum('abc,bdef,dghi,jgk->aehjcfik',
                               left_env, w1, w2, right_env, optimize='optimal')

        dim_L, d1, d2, dim_R = H_eff_flat.shape[0], H_eff_flat.shape[1], H_eff_flat.shape[2], H_eff_flat.shape[3]
        H_eff = H_eff_flat.reshape(dim_L * d1 * d2 * dim_R, dim_L * d1 * d2 * dim_R)

        # --- FIX: Add maxiter and ncv to handle difficult convergence ---
        # Increase iterations and the size of the Krylov subspace (ncv).
        # ncv must be > k, and is recommended to be > 2*k.
        try:
            eigvals, eigvecs = eigsh(H_eff, k=1, which='SA', maxiter=2000, ncv=30)
        except Exception as e:
            # Add a fallback to a full diagonalization if eigsh still fails.
            # This is much slower but robust.
            print("\nWarning: eigsh failed to converge, falling back to dense solver (eigh).")
            eigvals, eigvecs = np.linalg.eigh(H_eff)
            eigvals = eigvals[:1]
            eigvecs = eigvecs[:, :1]

        ground_state = eigvecs[:, 0].reshape(dim_L, d1, d2, dim_R)

        u, s, vh = np.linalg.svd(ground_state.transpose(1, 0, 2, 3).reshape(d1 * dim_L, d2 * dim_R),
                                 full_matrices=False)

        s, u, vh = s[:self.bond_dimension], u[:, :self.bond_dimension], vh[:self.bond_dimension, :]
        s /= np.linalg.norm(s)
        s_sqrt = np.sqrt(s)

        new_A = u.reshape(d1, dim_L, -1).transpose(1, 0, 2) * s_sqrt
        new_B = (vh.reshape(-1, d2, dim_R) * s_sqrt.reshape(-1, 1, 1)).transpose(0, 1, 2)

        return eigvals[0], new_A, new_B

    def _contract_env(self, mps_tensor, mpo_tensor, env_tensor, direction):
        if direction == 'left':
            new_env = np.einsum('abc,cde,bfgd,agh->efh',
                                env_tensor, mps_tensor, mpo_tensor, mps_tensor.conj(), optimize='optimal')
            return new_env
        else:  # direction == 'right'
            new_env = np.einsum('abc,dea,fbge,hgc->dfh',
                                env_tensor, mps_tensor, mpo_tensor, mps_tensor.conj(), optimize='optimal')
            return new_env


def _reconstruct_state_from_mps(mps: List[np.ndarray]) -> np.ndarray:
    """Reconstructs the full 2^N state vector from the MPS tensors."""
    state = mps[0]
    for i in range(1, len(mps)):
        state = np.tensordot(state, mps[i], axes=(-1, 0))
    # Final shape is (1, d1, d2, ..., dN, 1). Squeeze out boundaries.
    return state.flatten()


# In qtexture/validation/ising_model.py

def get_ising_ground_state(n_spins: int, h: float, j: float = 1.0, bond_dimension: int = 20,
                           num_sweeps: int = 10) -> QuantumState:
    """
    Finds the ground state of the 1D Ising model using the DMRG algorithm.

    Args:
        n_spins: The number of spins (qubits) in the chain.
        h: The strength of the transverse field.
        j: The coupling strength.
        bond_dimension: The maximum bond dimension for the MPS. Controls accuracy.
        num_sweeps: The number of DMRG sweeps to perform for convergence.

    Returns:
        A qtexture.QuantumState object representing the ground state density matrix.
    """
    # 1. Run DMRG to find the ground state in MPS format
    dmrg_solver = _DMRG(n_spins, bond_dimension, h, j)
    dmrg_solver.run(num_sweeps=num_sweeps)

    # 2. Reconstruct the full state vector from the MPS
    ground_state_vector = _reconstruct_state_from_mps(dmrg_solver.mps)

    # --- FIX: Explicitly re-normalize the state vector to correct for numerical errors ---
    norm = np.linalg.norm(ground_state_vector)
    if norm > 1e-9:  # Avoid division by zero for a potential zero vector
        ground_state_vector /= norm

    # 3. Construct the now-valid density matrix
    density_matrix = np.outer(ground_state_vector, ground_state_vector.conj())

    return QuantumState(density_matrix)