# qtexture: A Quantum State Texture Library 🔬

[![PyPI version](https://img.shields.io/pypi/v/qtexture.svg)](https://pypi.org/project/qtexture/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/pypi/pyversions/qtexture.svg)](https://pypi.org/project/qtexture)

`qtexture` is a lightweight, high-performance Python library for calculating resource monotones from the theory of **Quantum State Texture**. It provides a robust, numerically stable, and easy-to-use interface for researchers and students in quantum information science.

The library is a computational implementation of the concepts described in the paper *"Role of quantum state texture in probing resource theories and quantum phase transition"*.

***

## Key Features

* **Lightweight & Fast**: Built on a minimal `NumPy` & `SciPy` stack for easy installation and high performance.
* **Advanced Optimization**: Includes a powerful, quantum-inspired optimizer (`ProgQAOA`) to find basis-independent measures of texture that correspond to resources like entanglement.
* **Scientifically Validated**: The library's correctness is validated by reproducing a known quantum phase transition in the Transverse Field Ising Model.
* **Interoperable**: Provides optional, lightweight utilities to convert to and from objects in major frameworks like **Qiskit** and **QuTiP**.

***

## Library Validation
A comprehensive series of tests were done in order to validate the results of the library. Information about these tests can be found [here](VALIDATION_EXAMPLE.md).

## Installation

Installation instructions are provided for both regular users and developers.

### User Installation

If you want to use `qtexture` as a library in your own project, the recommended way to install it is with **pip**.

```bash
# Standard installation
pip install qtexture
```

### Developer Setup

If you want to contribute to the library, run the validation scripts, or work with the source code, setting up a dedicated Conda/Mamba environment is recommended.

1.  **Create the Environment**: Use the provided `environment.yml` file.
    ```bash
    mamba env create -f environment.yml
    ```
2.  **Activate the Environment**:
    ```bash
    conda activate qtexture-env
    ```
3.  **Install `qtexture`**: Install the library in "editable" mode so you can modify the source code. Then install the Metal Kernel.
    ```bash
    pip install -e .
    cd scripts
    ./build_metallib.sh
    ```
Note: Whenever any files in ``qtexture/kernels`` are modified, you must recompile them using Step 3. This will be built in ``build/lib.(your machine articheture)/qtexture/kernels``. Then copy ``_kernels.cpython-*.so`` into the ``./qtexture/kernels`` directory to use your recompiled source code.

***

## Quickstart: Basic Monotone Calculation

Here's how to calculate a simple, basis-dependent monotone for a GHZ state. The source paper establishes a direct link between texture and state purity, defining a monotone as "the difference between the maximum and minimum real parts of the density matrix elements".

The formula is given as:
$$M_P(\rho) = \max_{i,j}(\text{Re}(\rho_{ij})) - \min_{i,j}(\text{Re}(\rho_{ij}))$$

```python
import qtexture as qt

# 1. Create a 3-qubit Greenberger–Horne–Zeilinger (GHZ) state.
ghz_state = qt.states.create_ghz_state(num_qubits=3)

# 2. The QuantumState object validates its physical properties.
print(f"Is the state valid? {ghz_state.is_valid()}")
# >>> Is the state valid? True

# 3. Calculate the texture-based purity monotone in the computational basis.
purity_monotone = qt.calculate_purity_monotone(ghz_state)
print(f"Purity Monotone of the GHZ state: {purity_monotone}")
# >>> Purity Monotone of the GHZ state: 0.5
```

***

## Advanced Usage: Minimizing Texture with `ProgQAOA`

### What Does This Algorithm Achieve? (A Simple Explanation)

Imagine you have a complex, bumpy object, and you want to measure its "bumpiness." The bumpiness you measure depends on the angle from which you look at it. From one angle, it might look very rough, but from another, it might appear much smoother.

* **The Quantum State is the bumpy object.**
* **"Texture" is our measure of its "bumpiness."** This bumpiness represents a useful quantum property, like entanglement.
* **The "measurement basis" is the angle from which we look.**

The `ProgQAOA` optimizer intelligently explores all the different "angles" (measurement bases) to find the one that results in the **lowest possible texture**. This minimum value tells us about the fundamental, unchangeable nature of the quantum state, independent of how we choose to measure it.

### Scientific Motivation from the Source Paper

A "critical feature of this resource theory... is its inherent basis dependence". The amount of texture a state has is not an intrinsic property but is "defined relative to a chosen measurement basis". A new basis can be chosen by applying a unitary transformation, $\rho' = U \rho U^\dagger$.

To create intrinsic measures of quantum resources, the paper proposes a powerful strategy: define quantities by "minimizing the texture of a quantum state over a set of basis choices". For example, to quantify multipartite entanglement, the paper introduces "non-local texture," whose calculation "involves a minimization of the texture... over the set of all possible local measurement bases for each subsystem".

The `ProgQAOA` optimizer is the library's implementation of this crucial minimization procedure, enabling the calculation of these advanced, basis-independent quantities.

### Why a Custom Optimizer? A Comparison

The `prog_qaoa` optimizer was custom-built to outperform general-purpose libraries on its specific task. The key difference is **Specialization vs. Generality**.

| Feature | General QAOA Libraries (e.g., Qiskit)                                                                                             | Custom `prog_qaoa` in `qtexture`                                                                         |
| :--- |:----------------------------------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------|
| **🎯 Core Task** | Find the ground state of a Hamiltonian (general purpose).                                                                         | Minimize texture of a density matrix (highly specialized).                                               |
| **⚙️ How it Operates** | Evolves state vectors ($ \psi \rangle$).                                                                                          | Natively evolves **density matrices** ($\rho$). |
| **🧠 Built-in Heuristics** | Provides standard classical optimizers.                                                                                           | Includes custom heuristics like **Adaptive Layering** and **Freeze-then-Refine** for better convergence. |
| **📦 Dependencies** | Requires installing a large, full-stack framework.                                                                                | Has no heavy external dependencies, keeping the library lightweight.                                     |
| **🚀 Performance** | Good for general tasks.                                                                                                           | Can leverage custom C++ and GPU kernels for significant speedup on its specific task.                    |

### Example: Calculating the Intrinsic Texture of a W-State

This example calculates the minimal texture for a 3-qubit W-state, which quantifies its intrinsic multipartite correlations.

```python
import qtexture as qt
import numpy as np

# 1. Create the 3-qubit entangled W-state.
w_state = qt.states.create_w_state(num_qubits=3)

# 2. Define a "cost program" for the QAOA optimizer. This program
#    assigns a classical cost to each computational basis state.
#    Here, we use the parity of the bitstring (number of 1s).
def parity_program(bitstrings: np.ndarray) -> np.ndarray:
    # bitstrings is a (8, 3) array of all possible bit combinations.
    # We return a (8,) array of costs.
    return np.sum(bitstrings, axis=1)

# 3. Create a ProgramCost object. The optimizer will use this classical
#    program to build its quantum optimization routine.
cost_function = qt.ProgramCost(n_qubits=3, program=parity_program)

# 4. Run the minimization. This searches for the optimal basis transformation
#    that minimizes the purity monotone.
print("Optimizing W state to find its minimal texture...")
result = qt.minimize_texture(
    state=w_state,
    program_cost=cost_function,
    max_layers=4 # Controls the complexity of the search
)

min_texture = result.fun # The minimized value is in the .fun attribute

# 5. Compare the result to the original texture.
original_texture = qt.calculate_purity_monotone(w_state)
print("\n--- Comparison Results ---")
print(f"Original Texture (W-state): {original_texture:.4f}")
print(f"Minimal Texture (W-state):  {min_texture:.4f}")
```

### Full Optimizer Example with All Parameters

Here is a more advanced example demonstrating all available parameters to give you full control over the optimization process.

```python
# This example uses the same state and cost function as before.
# We are using SciPy's 'Nelder-Mead' for demonstration, but 'L-BFGS-B' is the default.

full_result = qt.minimize_texture_with_prog_qaoa(
    # --- Core Parameters ---
    state=w_state,
    program_cost=cost_function,
    
    # --- Optimization Control ---
    subsystems=[0, 2],    # Only optimize over qubits 0 and 2
    max_layers=10,        # Set a higher maximum number of QAOA layers
    tol_layer=1e-5,       # Stricter tolerance for adding new layers
    patience=2,           # Stop if no improvement after 2 new layers
    
    # --- Initialization ---
    init_mode="ramp",     # Use a 'ramp' initialization for parameters
    
    # --- Inner SciPy Optimizer Settings ---
    inner_method="Nelder-Mead", # Choose a different SciPy optimizer
    inner_options={"xatol": 1e-7, "fatol": 1e-7}, # Pass specific options to it
    
    # --- Reproducibility ---
    random_seed=42,       # Set a seed for reproducible random initializations
    
    # --- Hardware ---
    use_gpu=False         # Set to True to use GPU acceleration
)

print(f"\nMinimal Texture (Full Example): {full_result.fun:.4f}")
print(f"Optimization stopped at p={full_result.p} layers.")
```

### GPU Acceleration

For larger quantum systems, the optimization can be significantly accelerated by running it on a compatible GPU (e.g., Apple Silicon via Metal). This option is recommended only for very large quantum states (e.g., more than 10 qubits), where the computational cost on a CPU becomes a significant bottleneck. The overhead of transferring data to the GPU makes it less efficient for smaller systems.

To enable this, simply set the `use_gpu=True` flag.

**Important Note on Precision**: The GPU implementation achieves its speed by using single-precision floating-point numbers (`complex64`) instead of the double-precision (`complex128`) used by the CPU implementation. This trade-off means that while each optimization step is much faster, the lower precision can affect convergence. To reach the true global minimum with the same accuracy as the CPU version, you may need to increase both the number of layers (`max_layers`) and the layer tolerance (`tol_layer`).

```python
# Example of running the same optimization on a GPU with more layers
# and a stricter tolerance to compensate for the lower precision.

print("Optimizing W state on GPU...")
result_gpu = qt.minimize_texture_with_prog_qaoa(
    state=w_state,
    program_cost=cost_function,
    max_layers=8,      # Using more layers for the GPU run
    tol_layer=1e-8,    # Using a stricter tolerance
    use_gpu=True       # Enable GPU acceleration
)

min_texture_gpu = result_gpu.fun
print(f"Minimal Texture (W-state, GPU): {min_texture_gpu:.4f}")
```

***

## Contributing

Contributions are welcome! Please feel free to "report bugs, suggest features, or submit pull requests on the project's GitHub repository".

## Citing `qtexture`

If you use this library in your research, please cite the original paper that introduced the theory:

> *Role of quantum state texture in probing resource theories and quantum phase transition*, arXiv:2507.1382 [quant-ph]
