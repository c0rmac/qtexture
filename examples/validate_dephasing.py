# examples/validate_dephasing.py
import numpy as np
import matplotlib.pyplot as plt

# Imports from your library
from qtexture.states import create_ghz_state, create_maximally_mixed_state, QuantumState
from qtexture.prog_qaoa import minimize_texture, ProgramCost


def hamming_weight_cost(bitstrings):
    """A simple, generic cost function for the optimizer's program."""
    # bitstrings shape: (2^n, n)
    return bitstrings.sum(axis=1).astype(np.float64)


def main():
    """
    Validates that the minimized texture of a GHZ state approaches zero
    as it is dephased (mixed) with a maximally mixed state.
    """
    print("--- Validating Texture Minimization on a Dephased GHZ State ---")

    # --- 1. Setup ---
    num_qubits = 3
    print(f"System size: {num_qubits} qubits.")

    # Create the two states that define the endpoints of our mixing process.
    ghz_state = create_ghz_state(num_qubits)
    mixed_state = create_maximally_mixed_state(num_qubits)

    # Define the cost program for the optimizer.
    prog_cost = ProgramCost(n_qubits=num_qubits, program=hamming_weight_cost)

    # --- 2. Run the validation loop ---
    # Create a range of mixing parameters from 0 (pure) to 1 (fully mixed).
    mixing_params = np.linspace(0, 1, 15)
    minimized_texture_values = []

    print("\nCalculating minimized texture vs. dephasing parameter 'p'...")
    for i, p in enumerate(mixing_params):
        print(f"  Calculating for p = {p:.2f} ({i + 1}/{len(mixing_params)})")

        # Create the dephased state: ρ(p) = (1-p)ρ_GHZ + p*ρ_mixed
        dephased_rho_data = (1 - p) * ghz_state.data + p * mixed_state.data
        dephased_state = QuantumState(dephased_rho_data)

        # Run the optimizer to find the minimum possible texture for the current state.
        # Since we are minimizing texture over all qubits, the `subsystems` argument is not needed.
        result = minimize_texture(
            state=dephased_state,
            program_cost=prog_cost,
            max_layers=50,
            tol_layer=1e-4,
            inner_method="L-BFGS-B",
            inner_options={"maxiter": 300, "ftol": 1e-6},  # Relaxed tolerance for numerical stability,
            use_gpu=False
        )
        minimized_texture_values.append(result.fun)

    # --- 3. Plot the results ---
    plt.figure(figsize=(8, 5))
    plt.plot(mixing_params, minimized_texture_values, 'o-', label=f'Minimized Texture ({num_qubits}-qubit GHZ)')

    plt.xlabel("Dephasing Parameter (p)")
    plt.ylabel("Minimized Purity Monotone")
    plt.title("Validation: Minimized Texture vs. Dephasing")
    plt.legend()
    plt.grid(True)
    plt.ylim(bottom=-0.01)  # Ensure y=0 is clearly visible

    print("\nPlot generated. The graph should show a curve decreasing to zero as p approaches 1.")
    plt.show()


if __name__ == '__main__':
    main()