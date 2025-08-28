# examples/debug_gpu_parameters.py
import numpy as np

# Make sure your project is installed or the path is set correctly
from qtexture.states import QuantumState
from qtexture.prog_qaoa import ProgramBasedAnsatz, ProgramCost


def main():
    """
    This script tests if parameters sent to the GPU are received correctly.
    It requires the temporary modification of the 'qaoa_evolution_small_system'
    kernel in kernels.metal.
    """
    print("--- GPU Parameter Passing Debug Test ---")

    # 1. SETUP
    n_qubits = 2  # Use a small system to trigger the specialized kernel
    dim = 2 ** n_qubits

    # A dummy program cost is required by the ansatz constructor
    prog_cost = ProgramCost(n_qubits=n_qubits, c_vals=np.zeros(dim))

    # The ansatz object contains the apply_layers method we need to test
    ansatz = ProgramBasedAnsatz(n_qubits=n_qubits, program_cost=prog_cost)

    # Create a dummy initial state (must be complex64 to trigger the GPU path)
    rho0 = np.zeros((dim, dim), dtype=np.complex64)

    # 2. DEFINE TEST PARAMETERS
    # Define a unique, known value to send to the GPU
    test_beta_value = 0.12345
    betas = np.array([test_beta_value, 0.678], dtype=np.float32)
    gammas = np.zeros_like(betas)  # These don't matter for this test

    print(f"[*] Sending beta[0] = {test_beta_value} to the GPU kernel...")

    # 3. CALL THE KERNEL
    # This calls the Python -> C++ -> Metal toolchain
    result_rho = ansatz.apply_layers(rho0, betas, gammas)

    # 4. VERIFY THE RESULT
    # The modified kernel should have written beta[0] to rho[0][0].real
    # and our marker (999.0) to rho[0][0].imag
    value_read_back = result_rho[0, 0]
    print(f"[*] Value read back from rho[0,0]: {value_read_back}")

    expected_real = test_beta_value
    expected_imag = 999.0

    # Use np.isclose for safe floating point comparison
    if (np.isclose(value_read_back.real, expected_real) and
            np.isclose(value_read_back.imag, expected_imag)):
        print("\n✅ SUCCESS: Parameters are being passed to the GPU kernel correctly.")
    else:
        print(f"\n❌ FAILURE: Parameters are NOT being passed correctly.")
        print(f"  Expected: ({expected_real:.5f}, {expected_imag:.1f}j)")
        print(f"  Received: ({value_read_back.real:.5f}, {value_read_back.imag:.1f}j)")


if __name__ == "__main__":
    main()