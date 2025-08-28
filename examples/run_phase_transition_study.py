# examples/run_ising_validation.py

import numpy as np
import matplotlib.pyplot as plt

# Import the necessary functions from the qtexture library
from qtexture.validation.ising_model import get_ising_ground_state
from qtexture.monotones import calculate_purity_monotone


# In examples/run_ising_validation.py

def calculate_accuracy_metric(h_values: np.ndarray, texture_values: np.ndarray) -> dict:
    """
    Calculates how accurately the tool identifies the critical point, ignoring
    potential boundary effects.
    """
    h_critical_theoretical = 1.0
    d_texture_dh = np.gradient(texture_values, h_values)

    # --- MODIFICATION START ---

    # Define a search region to avoid boundary effects at h=0 and h=2.
    # We will only search for the peak in the central 80% of the data.
    num_points = len(h_values)
    start_index = int(num_points * 0.1)  # Ignore the first 10%
    end_index = int(num_points * 0.9)  # Ignore the last 10%

    # Find the index of the peak ONLY within this central region.
    search_region_derivative = np.abs(d_texture_dh[start_index:end_index])
    peak_index_in_region = np.argmax(search_region_derivative)

    # Convert the index back to its position in the original full array.
    peak_index = peak_index_in_region + start_index

    # --- MODIFICATION END ---

    h_critical_detected = h_values[peak_index]
    error = abs(h_critical_detected - h_critical_theoretical)

    return {
        "detected_critical_point": h_critical_detected,
        "theoretical_critical_point": h_critical_theoretical,
        "absolute_error": error
    }


def run_phase_transition_study(n_spins=10, h_values=np.linspace(0, 2, 50)):
    """
    Executes the validation study using the ground state solver.
    """
    texture_values = []
    print(f"Running validation for N={n_spins} spins...")
    for i, h in enumerate(h_values):
        print(f"  Calculating for h = {h:.3f} ({i + 1}/{len(h_values)})", end='\r')
        ground_state = get_ising_ground_state(
            n_spins=n_spins,
            h=h
        )
        texture = calculate_purity_monotone(ground_state)
        texture_values.append(texture)

    print("\n\nValidation run complete.")
    accuracy_results = calculate_accuracy_metric(h_values, np.array(texture_values))

    return h_values, np.array(texture_values), accuracy_results

def main():
    """Main function to run the study and plot results."""
    h_vals, tex_vals, accuracy = run_phase_transition_study()

    # Print the accuracy results to the console
    print("\n--- Accuracy Report ---")
    print(f"Theoretical Critical Point: {accuracy['theoretical_critical_point']:.4f}")
    print(f"Detected Critical Point:    {accuracy['detected_critical_point']:.4f}")
    print(f"Absolute Error:             {accuracy['absolute_error']:.4f}")
    print("-------------------------")

    # Plotting
    d_texture_dh = np.gradient(tex_vals, h_vals)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)

    ax1.plot(h_vals, tex_vals, 'o-', label='Purity Monotone')
    ax1.axvline(1.0, color='r', linestyle='--', label='Theoretical Critical Point (h=1.0)')
    ax1.set_ylabel("Texture (Purity Monotone)")
    ax1.set_title("Quantum State Texture at the Ising Model Phase Transition")
    ax1.grid(True)
    ax1.legend()

    ax2.plot(h_vals, d_texture_dh, 'o-', color='g', label='d(Texture)/dh')
    ax2.axvline(1.0, color='r', linestyle='--', label='Theoretical Critical Point (h=1.0)')
    ax2.axvline(accuracy['detected_critical_point'], color='b', linestyle=':',
                label=f"Detected Peak at h={accuracy['detected_critical_point']:.3f}")
    ax2.set_xlabel("Transverse Field Strength (h/J)")
    ax2.set_ylabel("Numerical Derivative of Texture")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()