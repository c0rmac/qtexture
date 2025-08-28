// qtexture/kernels/kernels.hpp
#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <string>

namespace py = pybind11;
using c64 = std::complex<double>;
using c32 = std::complex<float>;

// -------------------------
// CPU (always available)
// -------------------------

// Multiply in‑place by diag(phases) on left and conj(diag(phases)) on right
void apply_cost_phase_inplace(py::array_t<c64> rho,
                              py::array_t<c64> phases);

// Out‑of‑place 1‑qubit unitary application
py::array_t<c64> apply_1q_unitary_density(py::array_t<c64> rho,
                                          py::array_t<c64> U,
                                          int k, int n);

py::array_t<c64> evolve_all_layers_cpu(
    py::array_t<c64> rho0,
    py::array_t<double> betas,
    py::array_t<double> gammas,
    py::array_t<double> c_vals,
    py::array_t<int> active_qubits);

py::array_t<c32> evolve_all_layers_cpu_f32(
    py::array_t<c32> rho0,
    py::array_t<float> betas,
    py::array_t<float> gammas,
    py::array_t<float> c_vals,
    py::array_t<int> active_qubits);

// -------------------------
// Metal GPU (complex64 only)
// -------------------------

// Initialise Metal and load the compiled .metallib
void metal_init(const std::string& metallib_path);

// GPU cost‑phase for complex64
void apply_cost_phase_inplace_metal_f32(py::array_t<c32> rho,
                                        py::array_t<c32> phases);

// Single‑gate GPU path for complex64
void apply_1q_unitary_density_metal_f32(py::array_t<c32> rho,
                                        py::array_t<c32> U,
                                        int k, int n);

// Layer‑batched GPU path for complex64
// `qs` is a 1D int array of target qubit indices
void apply_1q_unitary_density_layer_metal_f32(py::array_t<c32> rho,
                                              py::array_t<c32> U,
                                              py::array qs,
                                              int n);

// Fused GPU path: phase + mixer layer in one command buffer
// qs is a 1D int array of target qubits
void apply_phase_and_mixer_layer_metal_f32(py::array_t<c32> rho,
                                           py::array_t<c32> phases,
                                           py::array_t<c32> U,
                                           py::array qs,
                                           int n);

// Fused: run all layers in one GPU command buffer (complex64 path)
void apply_layers_metal_f32(py::array_t<c32> rho,
                            py::array_t<float> betas,
                            py::array_t<float> gammas,
                            py::array_t<double> c_vals,  // ProgramCost.compile() is float64
                            py::array qs,                // 1D int array of active qubits
                            int n);

void apply_layers_small_system_metal_f32(py::array_t<c32> rho,
                                        py::array_t<float> betas,
                                        py::array_t<float> gammas,
                                        py::array_t<float> c_vals, // CORRECTED: float
                                        py::array qs,
                                        int n);

