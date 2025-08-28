import numpy as np
import qtexture as qt

# 1. Create a 3-qubit GHZ state using a library helper function [cite: 96]
ghz_state = qt.states.create_ghz_state(num_qubits=3)

# 2. The state object validates the physical properties of the density matrix
print(f"Is the state valid? {ghz_state.is_valid()}")
print(f"Trace of the state: {ghz_state.trace:.2f}")

# 3. Calculate the texture-based purity monotone [cite: 118]
purity_monotone = qt.calculate_purity_monotone(ghz_state)
print(f"Purity Monotone of the GHZ state: {purity_monotone}")

# 4. Change the basis of the state using a Hadamard gate on the first qubit
hadamard = (1 / np.sqrt(2)) * np.array([[1, 1], [1, -1]])
identity = np.eye(2)
# U = H ⊗ I ⊗ I
unitary_op = np.kron(hadamard, np.kron(identity, identity))

ghz_in_new_basis = qt.change_basis(ghz_state, unitary_op)

# 5. Calculate the monotone in the new basis [cite: 25]
purity_monotone_new_basis = qt.calculate_purity_monotone(ghz_in_new_basis)
print(f"Purity Monotone in the new basis: {purity_monotone_new_basis}")