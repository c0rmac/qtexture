# examples/validate_nonlocal_texture.py
import time

import numpy as np
import matplotlib.pyplot as plt

from qtexture import calculate_purity_monotone
from qtexture.prog_qaoa import minimize_texture, ProgramCost, ProgramBasedAnsatz
from qtexture.states import QuantumState
from qtexture.optimize import calculate_nonlocal_texture

# Replace with your custom texture-aware program as needed.
def hamming_weight_cost(bitstrings):
    # bitstrings: (2^n, n) uint8
    return bitstrings.sum(axis=1).astype(np.float64)

def create_tunable_state(theta: float) -> QuantumState:
    """Creates the state |ψ⟩ = cos(θ)|00⟩ + sin(θ)|11⟩."""
    psi = np.zeros(4, dtype=np.complex128)
    psi[0] = np.cos(theta)  # Amplitude for |00⟩
    psi[3] = np.sin(theta)  # Amplitude for |11⟩
    rho = np.outer(psi, psi.conj())
    return QuantumState(rho)


def run_performance_benchmark():
    print("\n[1] Testing separable state |00⟩ (θ=0) with basinhopping...")
    separable_state = create_tunable_state(0.0)

    times_new = []

    for i in range(50):
        dim = separable_state.data.shape[0]  # e.g., 4
        nq = int(np.log2(dim))  # 2 for your tunable state
        pcost = ProgramCost(n_qubits=nq, program=hamming_weight_cost)

        t2 = time.perf_counter()
        result2 = minimize_texture(
            state=separable_state,
            program_cost=pcost,
            subsystems=[0, 1],
            max_layers=4,
            tol_layer=1e-5,
            patience=1,
            inner_method="L-BFGS-B",
            inner_options={"maxiter": 300}
        )
        t3 = time.perf_counter()
        new_time = t3 - t2
        times_new.append(new_time)

        # --- Results ---
        #min_texture_separable = result.fun
        min_texture_separable2 = result2.fun

        print(f"[{i:02d}] New(QAOA): {min_texture_separable2:.6f} in {new_time:.4f}s")
        #print(f"     Old:       {min_texture_separable:.6f} in {old_time:.4f}s")
        assert np.isclose(min_texture_separable2, 0.0, atol=1e-4)
        print(f"✅ ({i}) Validation PASSED")

    # --- Summary stats ---
    print("\n=== Performance summary over 50 runs ===")
    #print(f"Old method avg time: {np.mean(times_old):.4f}s ± {np.std(times_old):.4f}s")
    print(f"New method avg time: {np.mean(times_new):.4f}s ± {np.std(times_new):.4f}s")
    #speedup = np.mean(times_old) / np.mean(times_new) if np.mean(times_new) > 0 else float('inf')
    #print(f"Speedup (old/new): {speedup:.2f}×")


def main():
    print("--- Validating calculate_nonlocal_texture (Simplified Call) ---")

    #run_performance_benchmark()

    # 2. Plot texture vs. entanglement to verify the peak
    print("\n[2] Plotting non-local texture vs. entanglement parameter θ...")
    thetas = np.linspace(0, np.pi / 2, 15)  # Range from separable to separable
    texture_values = []

    for i, theta in enumerate(thetas):
        print(f"  Calculating for θ = {theta:.2f} ({i + 1}/{len(thetas)})")
        state = create_tunable_state(theta)
        #res = calculate_nonlocal_texture(state, subsystems=[0, 1], method='basinhopping', niter=50)

        dim = state.data.shape[0]  # e.g., 4
        nq = int(np.log2(dim))  # 2 for your tunable state
        pcost = ProgramCost(n_qubits=nq, program=hamming_weight_cost)
        res = minimize_texture(
            state=state,
            program_cost=pcost,
            subsystems=[0, 1],  # or a list like [0, 2]
            max_layers=100,
            tol_layer=1e-5, # 1e-5
            patience=1,
            inner_method="L-BFGS-B",
            inner_options={"maxiter": 300},
            use_gpu=False
        )
        texture_values.append(res.fun)

    # Plotting the results
    plt.figure(figsize=(8, 5))
    plt.plot(thetas, texture_values, 'o-', label='Non-Local Texture')
    plt.axvline(np.pi / 4, color='r', linestyle='--', label='Max Entanglement (θ=π/4)')
    plt.xlabel("Entanglement Parameter θ")
    plt.ylabel("Non-Local Texture")
    plt.title("Validation of Non-Local Texture vs. Entanglement")
    plt.legend()
    plt.grid(True)
    print("\nPlot generated. Check if the peak aligns with the red dashed line.")
    plt.show()


if __name__ == '__main__':
    main()