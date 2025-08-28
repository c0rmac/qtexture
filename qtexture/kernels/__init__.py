# qtexture/kernels/__init__.py

"""
High-performance CPU and Metal GPU kernels for qtexture Prog-QAOA.

Wraps the compiled Pybind11 extension (_kernels) and re-exports
its functions at the package level.

Detects GPU (Metal) availability at import and provides:
    HAS_METAL flag,
    apply_1q_unitary_density_best(),
    apply_mixer_layer_best()
"""

from importlib import import_module
import pathlib
import numpy as np
import os

DEBUG = os.getenv("QTEXTURE_DEBUG", "0") == "1"
def _dbg(msg: str):
    if DEBUG:
        print(f"[qtexture.kernels] {msg}")

_k = import_module("qtexture.kernels._kernels")

# ----------------
# CPU entry points
# ----------------
# 1) diagonal cost-phase multiply
apply_cost_phase_inplace            = _k.apply_cost_phase_inplace
# 2) single-qubit mixer, in-place via Accelerate/BLAS
apply_1q_unitary_density_accel     = _k.apply_1q_unitary_density_inplace_accel
# ADDED: The new, fully optimized CPU evolution function
evolve_all_layers_cpu               = getattr(_k, "evolve_all_layers_cpu", None)

evolve_all_layers_cpu_f32 = getattr(_k, "evolve_all_layers_cpu_f32", None)

# ----------------
# GPU entry points (complex64 only)
# ----------------
HAS_METAL                             = False
apply_cost_phase_inplace_metal_f32    = getattr(_k, "apply_cost_phase_inplace_metal_f32", None)
apply_1q_unitary_density_metal_f32    = getattr(_k, "apply_1q_unitary_density_metal_f32", None)
apply_1q_unitary_density_layer_metal_f32 = getattr(
    _k, "apply_1q_unitary_density_layer_metal_f32", None
)
apply_phase_and_mixer_layer_metal_f32 = getattr(
    _k, "apply_phase_and_mixer_layer_metal_f32", None
)
apply_layers_metal_f32                = getattr(_k, "apply_layers_metal_f32", None)
apply_layers_small_system_metal_f32   = getattr(_k, "apply_layers_small_system_metal_f32", None)


# Try to init Metal if any GPU function is present
_metal_init = getattr(_k, "metal_init", None)
if _metal_init and any((
    apply_cost_phase_inplace_metal_f32,
    apply_1q_unitary_density_metal_f32,
    apply_1q_unitary_density_layer_metal_f32,
    apply_phase_and_mixer_layer_metal_f32,
    apply_layers_metal_f32,
    apply_layers_small_system_metal_f32
)):
    metallib_path = pathlib.Path(__file__).with_name("kernels.metallib")
    if metallib_path.exists():
        try:
            _metal_init(str(metallib_path))
            HAS_METAL = True
            _dbg(f"Metal initialised from {metallib_path}")
        except Exception as e:
            print(f"[qtexture] Metal init failed, falling back to CPU: {e}")

# ----------------
# Helpers
# ----------------
def apply_1q_unitary_density_best(rho, U, k, n):
    """
    Pick GPU if available (complex64), else CPU (complex128→BLAS).
    """
    if HAS_METAL and rho.dtype == np.complex64 and apply_1q_unitary_density_metal_f32:
        rho64 = np.ascontiguousarray(rho, dtype=np.complex64)
        U64   = np.ascontiguousarray(U,   dtype=np.complex64)
        _dbg(f"GPU mixer gate_f32 (k={k})")
        return apply_1q_unitary_density_metal_f32(rho64, U64, int(k), int(n))

    # CPU fallback (in-place BLAS)
    rho128 = np.ascontiguousarray(rho, dtype=np.complex128)
    U128   = np.ascontiguousarray(U,   dtype=np.complex128)
    _dbg(f"CPU mixer accel (k={k})")
    apply_1q_unitary_density_accel(rho128, U128, int(k), int(n))
    return rho128

def apply_mixer_layer_best(rho, U, active_qubits, n):
    """
    Apply a layer of single-qubit mixers on `active_qubits`.
    Uses the best available path: GPU-batched layer or CPU BLAS.
    """
    if HAS_METAL and rho.dtype == np.complex64 and apply_1q_unitary_density_layer_metal_f32:
        rho64 = np.ascontiguousarray(rho, dtype=np.complex64)
        U64   = np.ascontiguousarray(U,   dtype=np.complex64)
        qs    = np.ascontiguousarray(np.asarray(active_qubits, dtype=np.int32))
        _dbg(f"GPU mixer layer_f32 (qs={qs.tolist()})")
        try:
            apply_1q_unitary_density_layer_metal_f32(rho64, U64, qs, int(n))
            # copy back if needed
            if rho64 is not rho:
                rho[...] = rho64.astype(rho.dtype, copy=False)
            return
        except Exception as e:
            _dbg(f"GPU mixer layer failed: {e} -> CPU fallback")

    # CPU fallback: loop per qubit
    rho128 = np.ascontiguousarray(rho, dtype=np.complex128)
    U128   = np.ascontiguousarray(U,   dtype=np.complex128)
    for q in active_qubits:
        _dbg(f"CPU mixer accel (k={q})")
        apply_1q_unitary_density_accel(rho128, U128, int(q), int(n))
    return rho128


# Add a new "best" function to the Helpers section
def evolve_all_layers_cpu_best(rho0, betas, gammas, c_vals, active_qubits):
    """
    Evolves the state on the CPU using the kernel that matches the input dtype.
    Prevents unintended up-casting from complex64 to complex128.
    """
    if rho0.dtype == np.complex64:
        # --- Single Precision Path (complex64) ---
        if evolve_all_layers_cpu_f32 is None:
            raise RuntimeError("complex64 CPU kernel is not available.")

        # Ensure all inputs are the correct single-precision type
        rho_f32 = np.ascontiguousarray(rho0, dtype=np.complex64)
        betas_f32 = np.ascontiguousarray(betas, dtype=np.float32)
        gammas_f32 = np.ascontiguousarray(gammas, dtype=np.float32)
        c_vals_f32 = np.ascontiguousarray(c_vals, dtype=np.float32)
        active_qs_i32 = np.ascontiguousarray(active_qubits, dtype=np.int32)

        return evolve_all_layers_cpu_f32(rho_f32, betas_f32, gammas_f32, c_vals_f32, active_qs_i32)

    else:
        # --- Double Precision Path (complex128) ---
        if evolve_all_layers_cpu is None:
            raise RuntimeError("complex128 CPU kernel is not available.")

        # Ensure all inputs are the correct double-precision type
        rho_f64 = np.ascontiguousarray(rho0, dtype=np.complex128)
        betas_f64 = np.ascontiguousarray(betas, dtype=np.float64)
        gammas_f64 = np.ascontiguousarray(gammas, dtype=np.float64)
        c_vals_f64 = np.ascontiguousarray(c_vals, dtype=np.float64)
        active_qs_i32 = np.ascontiguousarray(active_qubits, dtype=np.int32)

        return evolve_all_layers_cpu(rho_f64, betas_f64, gammas_f64, c_vals_f64, active_qs_i32)


__all__ = [
    # CPU
    "apply_cost_phase_inplace",
    "apply_1q_unitary_density_accel",
    "evolve_all_layers_cpu", # ADDED: Export the new function
    # GPU direct
    "apply_cost_phase_inplace_metal_f32",
    "apply_1q_unitary_density_metal_f32",
    "apply_1q_unitary_density_layer_metal_f32",
    "apply_phase_and_mixer_layer_metal_f32",
    "apply_layers_metal_f32",
    "apply_layers_small_system_metal_f32",
    # Helpers
    "apply_1q_unitary_density_best",
    "apply_mixer_layer_best",
    # Capability flag
    "HAS_METAL",
]