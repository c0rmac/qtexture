# examples/compare_entangled_states.py (Updated to use ProgQAOA)

import numpy as np
from qtexture.states import create_ghz_state, create_w_state
# Updated import to use the custom ProgQAOA optimizer
from qtexture.prog_qaoa import minimize_texture, ProgramCost


def hamming_weight_cost(bitstrings: np.ndarray) -> np.ndarray:
    """
    A simple, generic cost function for the optimizer's program.
    It calculates the number of '1's in each bitstring.
    """
    return bitstrings.sum(axis=1).astype(np.float64)


def main():
    """
    This script uses the ProgQAOA optimizer to compare the intrinsic
    non-local texture of the GHZ and W states.
    """
    print("--- Comparing Non-Local Texture of 3-Qubit States using ProgQAOA ---")

    # 1. Create the quantum states
    num_qubits = 3
    ghz_state = create_ghz_state(num_qubits)
    w_state = create_w_state(num_qubits)

    # 2. Define the cost program for the optimizer. This program assigns a
    #    classical cost to each computational basis state. The optimizer
    #    uses this to construct its objective function.
    prog_cost = ProgramCost(n_qubits=num_qubits, program=hamming_weight_cost)

    print("\n[1] Analyzing the GHZ state...")
    print("Running ProgQAOA optimizer to find minimal texture...")
    # 3. Calculate the non-local texture for the GHZ state using the new optimizer.
    #    The optimization is performed over local unitaries on all 3 subsystems by default.
    ghz_result = minimize_texture(
        state=ghz_state,
        program_cost=prog_cost,
        max_layers=50,
        inner_options={"maxiter": 200, "ftol": 1e-6} # Options for SciPy's L-BFGS-B
    )
    ghz_min_texture = ghz_result.fun  # The minimized value is in the 'fun' attribute
    print(f"Optimization complete. Found minimal texture: {ghz_min_texture:.4f}")

    print("\n[2] Analyzing the W state...")
    print("Running ProgQAOA optimizer to find minimal texture...")
    # 4. Calculate the non-local texture for the W state.
    w_result = minimize_texture(
        state=w_state,
        program_cost=prog_cost,
        max_layers=50,
        inner_options={"maxiter": 200, "ftol": 1e-6}
    )
    w_min_texture = w_result.fun
    print(f"Optimization complete. Found minimal texture: {w_min_texture:.4f}")

    # 5. Compare the results
    print("\n--- Comparison Results ---")
    print(f"Non-Local Texture (GHZ): {ghz_min_texture:.4f}")
    print(f"Non-Local Texture (W):   {w_min_texture:.4f}")

    if np.isclose(ghz_min_texture, w_min_texture):
        print("\nThe states have a similar amount of non-local texture.")
    elif ghz_min_texture > w_min_texture:
        print("\nThe GHZ state has a higher non-local texture than the W state.")
    else:
        print("\nThe W state has a higher non-local texture than the GHZ state.")


if __name__ == '__main__':
    main()