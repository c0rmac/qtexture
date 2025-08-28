# qtexture/optimize.py

import numpy as np
from scipy.optimize import minimize, basinhopping, OptimizeResult
from .states import QuantumState
from .monotones import calculate_purity_monotone, change_basis


def create_su2_unitary(params: np.ndarray) -> np.ndarray:
    """
    Creates a single-qubit SU(2) unitary matrix from 3 real parameters (Euler angles).

    This is a helper function used by the objective function to parameterize
    the space of local unitaries.

    Args:
        params: A NumPy array [theta, phi, lambda].

    Returns:
        A 2x2 unitary NumPy array.
    """
    theta, phi, lam = params
    cos_t = np.cos(theta / 2)
    sin_t = np.sin(theta / 2)
    return np.array([
        [cos_t, -np.exp(1j * lam) * sin_t],
        [np.exp(1j * phi) * sin_t, np.exp(1j * (phi + lam)) * cos_t]
    ])


def _objective_function(params: np.ndarray, state: QuantumState, subsystems: list[int]) -> float:
    """
    Private objective function. It constructs a unitary, transforms the state,
    and returns the purity monotone. This is kept internal to the module.
    """
    num_qubits = int(np.log2(state.data.shape[0]))
    params_reshaped = params.reshape(-1, 3)

    unitary_list = []
    param_idx = 0
    for i in range(num_qubits):
        if i in subsystems:
            local_u = create_su2_unitary(params_reshaped[param_idx])
            param_idx += 1
        else:
            local_u = np.eye(2)
        unitary_list.append(local_u)

    full_unitary = unitary_list[0]
    for i in range(1, num_qubits):
        full_unitary = np.kron(full_unitary, unitary_list[i])

    transformed_state = change_basis(state, full_unitary)
    return calculate_purity_monotone(transformed_state)


def calculate_nonlocal_texture(
        state: QuantumState,
        subsystems: list[int],
        method: str = 'BFGS',
        **kwargs
) -> OptimizeResult:
    """
    Numerically finds the minimal texture by optimizing over local unitary bases.

    This function can use both local and global optimization methods, providing
    a flexible interface for both quick exploration and rigorous validation.

    Args:
        state: The initial multipartite quantum state.
        subsystems: A list of qubit indices for local unitaries.
        method: The optimization algorithm. Can be any method supported by
                scipy.optimize.minimize (e.g., 'BFGS', 'Nelder-Mead') or
                'basinhopping' for a global search.
        **kwargs: Additional keyword arguments to pass to the chosen optimizer
                  (e.g., niter=100 for basinhopping or tol=1e-8 for BFGS).

    Returns:
        The full OptimizeResult object from SciPy. The minimized value is
        accessible via the `.fun` attribute.
    """
    num_params = 3 * len(subsystems)
    initial_guess = np.random.rand(num_params) * 2 * np.pi
    args = (state, subsystems)

    if method.lower() == 'basinhopping':
        # Use the global basinhopping optimizer for a more robust search
        result = basinhopping(
            _objective_function,
            x0=initial_guess,
            minimizer_kwargs={'args': args},
            **kwargs
        )
    else:
        # Use a standard local optimizer from scipy.optimize.minimize
        result = minimize(
            _objective_function,
            x0=initial_guess,
            args=args,
            method=method,
            **kwargs
        )
    return result