from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py
import sys
import os
import shutil
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


# --- Custom build commands for Metal kernels ---
class build_ext_with_metal(_build_ext):
    """Custom build_ext to handle .mm files."""

    def build_extensions(self):
        if ".mm" not in self.compiler.src_extensions:
            self.compiler.src_extensions.append(".mm")
        super().build_extensions()


class build_metallib(_build_py):
    """Custom command to build the .metallib file and move it to the build dir."""

    def run(self):
        print("Building kernels.metallib...")
        kernels_dir = os.path.join(os.path.dirname(__file__), 'qtexture', 'kernels')
        metal_file = os.path.join(kernels_dir, 'kernels.metal')
        air_file = os.path.join(kernels_dir, 'kernels.air')
        metallib_file = os.path.join(kernels_dir, 'kernels.metallib')

        # Build the .metallib file in the source directory
        subprocess.run(['xcrun', '-sdk', 'macosx', 'metal', '-c', metal_file, '-o', air_file], check=True)
        subprocess.run(['xcrun', '-sdk', 'macosx', 'metallib', air_file, '-o', metallib_file], check=True)
        os.remove(air_file)

        print("Built kernels.metallib.")

        # Copy the built .metallib file to the temporary build directory
        build_lib_dir = os.path.join(self.build_lib, 'qtexture', 'kernels')
        os.makedirs(build_lib_dir, exist_ok=True)
        shutil.copy(metallib_file, build_lib_dir)
        print(f"Copied kernels.metallib to {build_lib_dir}.")

        # Call the original build_py command to continue the normal process
        _build_py.run(self)


# --- Pybind11 Extension Definition ---
compile_args = ["-O3", "-fvisibility=hidden", "-std=c++11", "-fobjc-arc"]
link_args = ["-framework", "Metal", "-framework", "Foundation"]

ext_modules = [
    Pybind11Extension(
        "qtexture.kernels._kernels",
        [
            "qtexture/kernels/bindings.cpp",
            "qtexture/kernels/cpu_kernels.cpp",
            "qtexture/kernels/metal_kernels.mm",
        ],
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

# --- Main setup function ---
setup(
    name="qtexture-kernels",
    version="0.1.0",
    packages=find_packages(include=["qtexture*"]),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": build_ext_with_metal,
        "build_py": build_metallib,
    },
    zip_safe=False,
)