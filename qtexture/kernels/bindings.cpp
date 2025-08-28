#include "kernels.hpp"  // forward declarations for all functions

PYBIND11_MODULE(_kernels, m) {
    m.doc() = "High‑performance CPU and Metal GPU kernels for qtexture Prog‑QAOA";

    // --- CPU ---
    m.def("apply_cost_phase_inplace", &apply_cost_phase_inplace);
    m.def("apply_1q_unitary_density_inplace_accel", &apply_1q_unitary_density);
    m.def("evolve_all_layers_cpu", &evolve_all_layers_cpu);
    m.def("evolve_all_layers_cpu_f32", &evolve_all_layers_cpu_f32, "Evolve a state (complex64) through all QAOA layers on the CPU.");


    // --- Metal init ---
    m.def("metal_init", &metal_init);

    // --- GPU Metal float32/complex64 ---
    m.def("apply_cost_phase_inplace_metal_f32", &apply_cost_phase_inplace_metal_f32);
    m.def("apply_1q_unitary_density_metal_f32", &apply_1q_unitary_density_metal_f32);
    m.def("apply_1q_unitary_density_layer_metal_f32", &apply_1q_unitary_density_layer_metal_f32);
    m.def("apply_phase_and_mixer_layer_metal_f32", &apply_phase_and_mixer_layer_metal_f32);
    m.def("apply_layers_metal_f32", &apply_layers_metal_f32);
    m.def("apply_layers_small_system_metal_f32", &apply_layers_small_system_metal_f32);
}
