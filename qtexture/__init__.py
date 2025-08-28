# qtexture/__init__.py

"""
qtexture: A library for calculating quantum state texture monotones.
"""

# Expose the core classes and functions at the top level of the package
# as proposed in the API reference[cite: 252].
from .states import QuantumState
from .monotones import calculate_purity_monotone, change_basis
from .optimize import calculate_nonlocal_texture

# Make submodules accessible
from . import states
from . import utils
from . import validation