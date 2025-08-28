# qtexture/prog_qaoa.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Dict, Any, List, Tuple
import numpy as np
from scipy.optimize import minimize, basinhopping, OptimizeResult

from .states import QuantumState
from .monotones import calculate_purity_monotone

# ========= Utilities: basis, cost compilation, and single-qubit application =========

# ========= Fast bitstrings with caching =========

_BITSTRINGS_CACHE: Dict[int, np.ndarray] = {}

def _bitstrings(n: int) -> np.ndarray:
    """
    Vectorized bitstring enumeration (LSB-first), cached per n.
    Shape: (2^n, n), dtype=uint8.
    """
    if n in _BITSTRINGS_CACHE:
        return _BITSTRINGS_CACHE[n]
    N = 1 << n
    # Bits via shifts; LSB-first across columns
    bs = ((np.arange(N, dtype=np.uint64)[:, None] >> np.arange(n, dtype=np.uint64)) & 1).astype(np.uint8)
    _BITSTRINGS_CACHE[n] = bs
    return bs

def _int_to_bits(i: int, n: int) -> np.ndarray:
    return np.fromiter(((i >> k) & 1 for k in range(n)), count=n, dtype=np.uint8)

def _enumerate_bitstrings(n: int) -> np.ndarray:
    # Shape: (2^n, n), LSB-first; cheap and deterministic
    N = 1 << n
    out = np.empty((N, n), dtype=np.uint8)
    for i in range(N):
        out[i] = _int_to_bits(i, n)
    return out

def _compile_program_cost(n_qubits: int, program: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Vectorizes the classical 'program' c(x) over all computational basis states.

    program: accepts an array of bitstrings with shape (2^n, n) and returns
             a 1D array of costs with shape (2^n,).
    Returns: c_vals of shape (2^n,), dtype float64
    """
    bitstrings = _enumerate_bitstrings(n_qubits)
    c_vals = np.asarray(program(bitstrings), dtype=np.float64)
    if c_vals.shape != (1 << n_qubits,):
        raise ValueError("Program must return a vector of length 2^n.")
    return c_vals

def _apply_diagonal_phase_to_density(rho: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """
    Applies ρ -> D ρ D† where D = diag(phases).
    rho: (2^n, 2^n), phases: (2^n,) complex phases on the computational basis.
    This uses broadcasting — no large unitary is formed.
    """
    return (phases[:, None] * rho) * np.conj(phases)[None, :]

def _rx(theta: float) -> np.ndarray:
    # Rx(theta) = exp(-i theta X / 2) = [[cos(theta/2), -i sin(theta/2)],
    #                                    [-i sin(theta/2), cos(theta/2)]]
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    return np.array([[c, -1j * s],
                     [-1j * s, c]], dtype=np.complex128)

def _apply_1q_unitary_to_density(rho: np.ndarray, U: np.ndarray, k: int, n: int, dims: Tuple[int, ...]) -> np.ndarray:
    """
    In-place style application: ρ -> (U_k) ρ (U_k)† on qubit k of an n-qubit density matrix.
    Efficient via reshape + tensordot without forming Kronecker products.
    """
    # Reshape to 2n-index tensor
    rho_t = rho.reshape(dims + dims)

    # Apply on ket index k
    rho_t = np.tensordot(U, rho_t, axes=([1], [k]))  # new axis 0 inserted
    rho_t = np.moveaxis(rho_t, 0, k)

    # Apply on bra index n + k with U.conj()
    Uc = U.conj()
    rho_t = np.tensordot(Uc, rho_t, axes=([1], [n + k]))
    rho_t = np.moveaxis(rho_t, 0, n + k)

    return rho_t.reshape(rho.shape)


# ========= Program-based ansatz =========

@dataclass
class ProgramCost:
    """
    Program-based cost oracle c(x).
    Provide either:
      - program: callable mapping (2^n, n) bitstrings -> (2^n,) costs
    or:
      - c_vals: precomputed costs of length 2^n.

    normalize=True scales max|c| to 1 for stable γ scaling.
    """
    n_qubits: int
    program: Optional[Callable[[np.ndarray], np.ndarray]] = None
    c_vals: Optional[np.ndarray] = None
    normalize: bool = True

    # Internal cache
    _compiled: Optional[np.ndarray] = None
    _scale: float = 1.0

    def compile(self) -> np.ndarray:
        if self._compiled is not None:
            return self._compiled

        if self.c_vals is not None:
            c = np.asarray(self.c_vals, dtype=np.float64)
            if c.shape != (1 << self.n_qubits,):
                raise ValueError("c_vals must have length 2^n.")
        else:
            if self.program is None:
                c = np.zeros(1 << self.n_qubits, dtype=np.float64)
            else:
                bs = _bitstrings(self.n_qubits)
                c = np.asarray(self.program(bs), dtype=np.float64)
                if c.shape != (1 << self.n_qubits,):
                    raise ValueError("Program must return a vector of length 2^n.")

        if self.normalize:
            s = float(np.max(np.abs(c)))
            if s > 0:
                c = c / s
                self._scale = s
            else:
                self._scale = 1.0
        else:
            self._scale = 1.0

        self._compiled = c
        return c

    @property
    def scale(self) -> float:
        # 1.0 for zero/normalized costs, else original max-abs scale
        return self._scale


class ProgramBasedAnsatz:
    """
    Alternating layers: U(θ) = ∏_{l=1..p} [ U_M(β_l) · U_C(γ_l) ], right-applied to ρ.

    - Cost layer U_C(γ) = diag( exp(-i γ c(x)) ) compiled from classical program c(x).
    - Mixer U_M(β) = ∏_i exp(-i β X_i) = ⊗_i Rx_i(2β).

    Supports limiting action to a subset of qubits via 'subsystems' (indices).
    """
    def __init__(self, n_qubits: int, program_cost: ProgramCost, subsystems: Optional[Sequence[int]] = None):
        self.n = n_qubits
        self.c_vals = program_cost.compile()  # cached
        # Subsystem mask and indices
        if subsystems is None:
            self._mask = np.ones(self.n, dtype=bool)
        else:
            mask = np.zeros(self.n, dtype=bool)
            mask[list(subsystems)] = True
            self._mask = mask
        self._active = np.nonzero(self._mask)[0]
        # Precompute reshape dims tuple once
        self._dims = tuple(2 for _ in range(self.n))

    def apply_layers(self, rho0: np.ndarray, betas: np.ndarray, gammas: np.ndarray, use_gpu: bool) -> np.ndarray:
        if betas.shape != gammas.shape:
            raise ValueError("betas and gammas must match length p.")

        from qtexture import kernels as K
        dim = rho0.shape[0]
        GPU_DIM_THRESHOLD = 4096  # tune experimentally

        #use_gpu = (
        #        K.HAS_METAL
        #        and hasattr(K, "apply_layers_metal_f32")
        #        and dim >= GPU_DIM_THRESHOLD
        #)

        #use_gpu = True
        use_small_solution = False

        dt = np.complex64 if use_gpu else np.complex128
        #rho = np.ascontiguousarray(rho0, dtype=dt)

        if use_gpu and use_small_solution and dim <= 64:  # Use specialized kernel for n <= 6 qubits
            rho = np.ascontiguousarray(rho0, dtype=np.complex64)
            n_qubits = int(np.log2(dim))

            K.apply_layers_small_system_metal_f32(
                rho,
                np.asarray(betas, dtype=np.float32),
                np.asarray(gammas, dtype=np.float32),
                np.asarray(self.c_vals, dtype=np.float32),
                np.asarray(self._active, dtype=np.int32),
                n_qubits
            )
            return rho.astype(dt, copy=False)

        elif use_gpu:  # Use original kernel for n > 6 qubits
            # Run the entire stack on GPU with a single wait
            betas = np.asarray(betas, dtype=np.float32, order="C")
            gammas = np.asarray(gammas, dtype=np.float32, order="C")
            c_vals = np.asarray(self.c_vals, dtype=np.float64, order="C")
            qs = np.asarray(self._active, dtype=np.int32)

            rho = np.ascontiguousarray(rho0, dtype=dt)
            # rho = rho0.copy()
            K.apply_layers_metal_f32(rho, betas, gammas, c_vals, qs, int(self.n))
            return rho

        # Ensure types are correct for the C++ extension
        return K.evolve_all_layers_cpu_best(
                rho0,
                betas,
                gammas,
                self.c_vals,
                self._active
        )

# ========= Optimizer with adaptive layering =========

@dataclass
class ProgQAOAResult:
    fun: float
    x: np.ndarray
    p: int
    history: List[Dict[str, Any]]
    message: str
    success: bool
    rho_history: List[float]

class ProgQAOAOptimizer:
    """
    Adaptive layering with robust defaults:
      - Identity baseline guard
      - Near-zero deterministic init
      - Tiny-grid multi-start for new layer
      - Freeze-then-refine growth
    """
    def __init__(
        self,
        objective: Optional[Callable[[QuantumState], float]] = None,
        max_layers: int = 6,
        tol_layer: float = 1e-4,
        patience: int = 1,
        inner_optimizer: Optional[Any] = None,  # ACCEPTS AN OPTIMIZER OBJECT
        inner_method: str = "L-BFGS-B",  # Fallback for SciPy

        inner_options: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        identity_baseline: bool = True,
        init_mode: str = "near_zero",
        init_scale: float = 5e-3,
        multistart_grid: Sequence[float] = (-2e-2, 0.0, 2e-2),
        freeze_then_refine: bool = True,
        use_gpu: bool = False,
    ):
        self.objective = objective or calculate_purity_monotone
        self.max_layers = max_layers
        self.tol_layer = tol_layer
        self.patience = patience
        self.inner_optimizer = inner_optimizer  # STORE THE OPTIMIZER
        self.inner_method = inner_method
        self.inner_options = inner_options or {"maxiter": 500, "ftol": 1e-12}
        self.rng = np.random.default_rng(random_seed)
        self.identity_baseline = identity_baseline
        self.init_mode = init_mode
        self.init_scale = init_scale
        self.multistart_grid = tuple(multistart_grid)
        self.freeze_then_refine = freeze_then_refine
        self.use_gpu = use_gpu
        self.rho_history = []

    @staticmethod
    def _pack(betas: np.ndarray, gammas: np.ndarray) -> np.ndarray:
        return np.concatenate([betas, gammas])

    @staticmethod
    def _unpack(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        half = x.size // 2
        return x[:half], x[half:]

    def _objective_wrapped(self, x: np.ndarray, ansatz: ProgramBasedAnsatz, rho0: np.ndarray) -> float:
        betas, gammas = self._unpack(x)
        rho = ansatz.apply_layers(rho0, betas, gammas, self.use_gpu)
        return self.objective(QuantumState(rho))

    def _bounds(self, p: int) -> List[Tuple[float, float]]:
        return [(-np.pi, np.pi)] * (2 * p)

    def _minimize(self, fun, x0, bounds, args) -> OptimizeResult:
        """Internal dispatcher for the optimization backend."""
        # --- NEW: LOGIC TO USE A CUSTOM OPTIMIZER LIKE SPSA ---
        if self.inner_optimizer is not None:
            # Qiskit optimizers expect a function with a single argument (the parameters)
            wrapped_fun = lambda x: fun(x, *args)

            # Use the minimize method of the provided optimizer object
            qiskit_result = self.inner_optimizer.minimize(
                fun=wrapped_fun,
                x0=x0,
                bounds=np.array(bounds)
            )
            # Adapt the output to a SciPy-like result object for compatibility
            return OptimizeResult(
                fun=qiskit_result.fun,
                x=qiskit_result.x,
                nfev=getattr(qiskit_result, 'nfev', 0),
                message="Optimization with custom optimizer terminated."
            )

        # --- Original SciPy Logic ---
        if self.inner_method.lower() == "basinhopping":
            minimizer_kwargs = {
                "method": "L-BFGS-B", "args": args, "bounds": bounds, "options": self.inner_options
            }
            return basinhopping(func=fun, x0=x0, minimizer_kwargs=minimizer_kwargs, niter=50)
        else:
            return minimize(fun=fun, x0=x0, args=args, method=self.inner_method, options=self.inner_options,
                            bounds=bounds)

    def _init_params(self, p: int) -> np.ndarray:
        if self.init_mode == "near_zero":
            betas = np.full(p, self.init_scale)
            gammas = np.full(p, self.init_scale)
        elif self.init_mode == "ramp":
            t = np.linspace(0.0, 1.0, p)
            betas = self.init_scale * (0.5 + 0.5 * t)
            gammas = self.init_scale * (0.5 + 0.5 * (1.0 - t))
        else:  # "small_random"
            betas = self.rng.normal(scale=self.init_scale, size=p)
            gammas = self.rng.normal(scale=self.init_scale, size=p)
        return self._pack(betas, gammas)

    def _identity_value(self, state: QuantumState) -> float:
        return self.objective(state)

    def fit(
        self,
        state: QuantumState,
        program_cost: ProgramCost,
        subsystems: Optional[Sequence[int]] = None,
        init_p: int = 1,
    ) -> ProgQAOAResult:
        n = int(np.log2(state.data.shape[0]))
        ansatz = ProgramBasedAnsatz(n_qubits=n, program_cost=program_cost, subsystems=subsystems)
        rho0 = state.data

        history: List[Dict[str, Any]] = []
        fail_count = 0

        # Identity baseline floor
        floor_val = self._identity_value(state) if self.identity_baseline else np.inf
        best_val = floor_val
        best_x = np.zeros(2 * init_p, dtype=np.float64)

        # Initial optimize at p=init_p
        p = init_p
        x0 = self._init_params(p)
        res0 = self._minimize(self._objective_wrapped, x0, self._bounds(p), args=(ansatz, rho0))
        cand_val, cand_x = float(res0.fun), res0.x
        if cand_val < best_val:
            best_val, best_x = cand_val, cand_x
        history.append({"p": p, "fun": float(min(cand_val, floor_val)), "message": res0.message})

        # Adaptive growth
        while p < self.max_layers:
            p_new = p + 1
            betas, gammas = self._unpack(best_x)
            best_local_val = np.inf
            best_local_x = None

            for b0 in self.multistart_grid:
                for g0 in self.multistart_grid:
                    x_new0 = self._pack(np.concatenate([betas, [b0]]), np.concatenate([gammas, [g0]]))

                    if self.freeze_then_refine:
                        # Optimize last two params first
                        def fun_free(x_last2):
                            x_full = x_new0.copy()
                            x_full[-2:] = x_last2
                            return self._objective_wrapped(x_full, ansatz, rho0)

                        res_free = self._minimize(fun_free, x_new0[-2:], self._bounds(1), args=())
                        x_mid = x_new0.copy()
                        x_mid[-2:] = res_free.x
                        # Short global refine
                        res_ref = self._minimize(self._objective_wrapped, x_mid, self._bounds(p_new), args=(ansatz, rho0))
                        val, x_cand = float(res_ref.fun), res_ref.x
                    else:
                        res_ref = self._minimize(self._objective_wrapped, x_new0, self._bounds(p_new), args=(ansatz, rho0))
                        val, x_cand = float(res_ref.fun), res_ref.x

                    if val < best_local_val:
                        best_local_val, best_local_x = val, x_cand

            history.append({"p": p_new, "fun": best_local_val, "message": "grid+refine"})

            imp = best_val - best_local_val
            if imp >= self.tol_layer:
                p = p_new
                best_val = best_local_val
                best_x = best_local_x
                fail_count = 0
            else:
                fail_count += 1
                if fail_count > self.patience:
                    break

        final_val = min(best_val, floor_val) if self.identity_baseline else best_val
        msg = f"Stopped at p={p} with best fun={final_val:.6g}"
        return ProgQAOAResult(fun=float(final_val), x=best_x, p=p, history=history, message=msg, success=True, rho_history=self.rho_history)


# ========= Convenience wrapper for texture minimization =========

def minimize_texture(
    state: QuantumState,
    program_cost: ProgramCost,
    subsystems: Optional[Sequence[int]] = None,
    max_layers: int = 6,
    tol_layer: float = 1e-4,
    patience: int = 1,
    inner_optimizer: Optional[Any] = None,  # ACCEPTS AN OPTIMIZER OBJECT
    inner_method: str = "L-BFGS-B",  # Fallback for SciPy
    inner_options: Optional[Dict[str, Any]] = None,
    random_seed: Optional[int] = None,
    init_mode: str = "small_random",
    use_gpu: bool = False,
) -> ProgQAOAResult:
    """
    Minimize the texture-based purity monotone using Program-Based QAOA with adaptive layering.

    Built-in robustness and low-level optimisations:
      - Identity baseline floor
      - Near-zero deterministic init
      - Tiny-grid multistart & freeze-then-refine for layer growth
      - Cached bitstrings and program cost
      - In-place cost phases, precomputed dims/active indices
    """
    # Infer n and ensure ProgramCost matches
    dim = state.data.shape[0]
    n_from_state = int(np.log2(dim))
    if (1 << n_from_state) != dim:
        raise ValueError("State dimension must be a power of 2.")

    if getattr(program_cost, "n_qubits", None) != n_from_state:
        if getattr(program_cost, "program", None) is not None and getattr(program_cost, "c_vals", None) is None:
            program_cost = ProgramCost(
                n_qubits=n_from_state,
                program=program_cost.program,
                normalize=getattr(program_cost, "normalize", True),
            )
        else:
            raise ValueError(
                f"ProgramCost.n_qubits={getattr(program_cost, 'n_qubits', None)} "
                f"does not match state qubits {n_from_state}. "
                "Provide a program (not precomputed c_vals) or rebuild ProgramCost."
            )

    # Merge inner options
    inner_opts = {"maxiter": 500, "ftol": 1e-12}
    if inner_options:
        inner_opts.update(inner_options)

    opt = ProgQAOAOptimizer(
        objective=calculate_purity_monotone,
        max_layers=max_layers,
        tol_layer=tol_layer,
        patience=patience,
        inner_optimizer=inner_optimizer,
        inner_method=inner_method,
        inner_options=inner_opts,
        random_seed=random_seed,
        identity_baseline=True,
        init_mode=init_mode,
        init_scale=5e-3,
        multistart_grid=(-2e-2, 0.0, 2e-2),
        freeze_then_refine=True,
        use_gpu=use_gpu
    )
    return opt.fit(
        state=state,
        program_cost=program_cost,
        subsystems=subsystems,
        init_p=1,
    )