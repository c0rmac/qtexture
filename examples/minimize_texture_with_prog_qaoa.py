# examples/validate_phase_transition_optimizer.py
import time

import numpy as np
import matplotlib.pyplot as plt

from qtexture.states import QuantumState
from qtexture.prog_qaoa import minimize_texture, ProgramCost
from qtexture.validation.ising_model import get_ising_ground_state

def hamming_weight_cost(bitstrings):
    """A simple, generic cost function for the optimizer's program."""
    return bitstrings.sum(axis=1).astype(np.float64)


def main():
    """
    Validates that minimize_texture_with_prog_qaoa can detect the quantum
    phase transition in the Transverse Field Ising Model.
    """
    print("--- Validating Optimizer on a Quantum Phase Transition ---")

    # --- 1. Setup ---
    n_spins = 10  # NOTE: This is computationally intensive. 6-8 spins is a good range.
    print(f"System size: {n_spins} spins.")

    prog_cost = ProgramCost(n_qubits=n_spins, program=hamming_weight_cost)
    # Optimize over all local unitaries by specifying all subsystems
    all_qubits = list(range(n_spins))

    h_values = np.linspace(0.0, 2.0, 50)  # Reduced points due to higher cost
    minimized_textures = []

    # --- 2. Run the validation loop ---
    print("\nCalculating minimized (non-local) texture vs. field strength 'h'...")
    for i, h in enumerate(h_values):
        print(f"  Calculating for h = {h:.3f} ({i + 1}/{len(h_values)})")

        ground_state = get_ising_ground_state(n_spins, h)

        t2 = time.perf_counter()
        # Use the optimizer to find the minimal texture over local bases
        result = minimize_texture(
            state=ground_state,
            program_cost=prog_cost,
            subsystems=all_qubits,
            max_layers=100,  # Keep layers low to manage runtime
            patience=0,  # Stop adaptive layering early
            inner_options={"maxiter": 150, "ftol": 1e-5},
            use_gpu=True
        )
        t1 = time.perf_counter()
        dt = t1 - t2
        print(f"Time elapsed: {dt:.3f}s")
        print(f"Num layers: {result.p}")

        minimized_textures.append(result.fun)

    # --- 3. Plot the results ---
    d_texture_dh = np.gradient(minimized_textures, h_values[1] - h_values[0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    ax1.plot(h_values, minimized_textures, 'o-', label='Minimized (Non-Local) Texture')
    ax1.axvline(x=1.0, color='r', linestyle='--', label='Known Critical Point (h=1)')
    ax1.set_ylabel("Minimized Purity Monotone")
    ax1.set_title(f"Non-Local Texture of the Ising Ground State ({n_spins} spins)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(h_values, d_texture_dh, 'o-', color='g', label='Derivative of Texture')
    ax2.axvline(x=1.0, color='r', linestyle='--', label='Known Critical Point (h=1)')
    ax2.set_xlabel("Transverse Field Strength (h)")
    ax2.set_ylabel("d(Texture)/dh")
    ax2.legend()
    ax2.grid(True)

    fig.tight_layout()
    print("\nPlot generated. The derivative should show a peak near h=1.")
    plt.show()


if __name__ == '__main__':
    main()