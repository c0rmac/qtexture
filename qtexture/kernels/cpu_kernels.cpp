// qtexture/kernels/cpu_kernels.cpp
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <stdexcept>
#include <Accelerate/Accelerate.h>

namespace py = pybind11;
using c64 = std::complex<double>;

// cheap checks
static void ensure_c_contiguous(const py::array& a) {
  if (!(a.flags() & py::array::c_style))
    throw std::runtime_error("array must be C-contiguous");
}
static void ensure_c64(const py::array& a) {
  if (!py::isinstance<py::array_t<c64>>(a))
    throw std::runtime_error("array must be complex128");
}

// ρ[i,j] *= phases[i] * conj(phases[j]) (in-place)
void apply_cost_phase_inplace(py::array_t<c64> rho,
                              py::array_t<c64> phases)
{
  ensure_c_contiguous(rho); ensure_c64(rho);
  ensure_c_contiguous(phases); ensure_c64(phases);

  auto Rb = rho.request(), Pb = phases.request();
  ssize_t dim = Rb.shape[0];
  auto *R = (c64*)Rb.ptr, *P = (c64*)Pb.ptr;

  for (ssize_t i = 0; i < dim; ++i) {
    c64 pi = P[i];
    for (ssize_t j = 0; j < dim; ++j) {
      R[i*dim + j] *= pi * std::conj(P[j]);
    }
  }
}

// Helpers
static void ensure_c_contiguous(const py::array& arr, const char* name) {
    if (!(arr.flags() & py::array::c_style)) {
        throw std::runtime_error("rho must be C-contiguous");
    }
}
static void ensure_dtype_complex128(const py::array& arr, const char* name) {
    if (!py::isinstance<py::array_t<c64>>(arr)) {
        throw std::runtime_error(std::string(name) + " must be complex128");
    }
}

// ρ' = (U_k) ρ (U_k)†, where U is 2x2 unitary, k is target qubit (0-based), n is total qubits.
// Returns a new array; Python wrapper can overwrite input if desired.
py::array_t<c64> apply_1q_unitary_density(py::array_t<c64> rho, py::array_t<c64> U, int k, int n) {
    ensure_c_contiguous(rho, "rho");
    ensure_c_contiguous(U, "U");
    ensure_dtype_complex128(rho, "rho");
    ensure_dtype_complex128(U, "U");

    auto buf_rho = rho.request();
    auto buf_U = U.request();
    if (buf_rho.ndim != 2) throw std::runtime_error("rho must be 2D");
    if (buf_U.ndim != 2) throw std::runtime_error("U must be 2D");
    if (buf_U.shape[0] != 2 || buf_U.shape[1] != 2) throw std::runtime_error("U must be 2x2");

    ssize_t dim = buf_rho.shape[0];
    if (buf_rho.shape[1] != dim) throw std::runtime_error("rho must be square");
    if ((1LL << n) != dim) throw std::runtime_error("dim must equal 2^n");
    if (k < 0 || k >= n) throw std::runtime_error("k out of range");

    const auto* R = static_cast<const c64*>(buf_rho.ptr);
    const auto* UU = static_cast<const c64*>(buf_U.ptr);

    py::array_t<c64> out({dim, dim});
    auto buf_out = out.request();
    auto* O = static_cast<c64*>(buf_out.ptr);
    std::fill_n(O, dim * dim, c64(0.0, 0.0));

    // Preload U and U†
    c64 U00 = UU[0], U01 = UU[1], U10 = UU[2], U11 = UU[3];
    c64 V00 = std::conj(UU[0]), V01 = std::conj(UU[2]), V10 = std::conj(UU[1]), V11 = std::conj(UU[3]);

    const ssize_t bit = (1LL << k);

    for (ssize_t i = 0; i < dim; ++i) {
        int bi = (i & bit) ? 1 : 0;
        ssize_t base_i = i & ~bit;
        for (ssize_t j = 0; j < dim; ++j) {
            int bj = (j & bit) ? 1 : 0;
            ssize_t base_j = j & ~bit;

            // Accumulate over ai, aj in {0,1}
            // new_rho[ii,jj] += U[ai,bi] * rho[i,j] * conj(U[aj,bj])
            c64 a = R[i*dim + j];

            // Unroll 4 contributions:
            // (ai=0,aj=0)
            O[(base_i | 0)*dim + (base_j | 0)] += (bi==0?U00:U01) * a * (bj==0?V00:V10);
            // (ai=0,aj=1)
            O[(base_i | 0)*dim + (base_j | bit)] += (bi==0?U00:U01) * a * (bj==0?V01:V11);
            // (ai=1,aj=0)
            O[(base_i | bit)*dim + (base_j | 0)] += (bi==0?U10:U11) * a * (bj==0?V00:V10);
            // (ai=1,aj=1)
            O[(base_i | bit)*dim + (base_j | bit)] += (bi==0?U10:U11) * a * (bj==0?V01:V11);
        }
    }
    return out;
}

// Evolves a state through all p layers of the QAOA ansatz in a single call.
py::array_t<c64> evolve_all_layers_cpu(
    py::array_t<c64> rho0,
    py::array_t<double> betas,
    py::array_t<double> gammas,
    py::array_t<double> c_vals,
    py::array_t<int> active_qubits)
{
    // 1. --- Input Validation and Buffer Setup ---
    auto buf_rho0 = rho0.request();
    auto buf_betas = betas.request();
    auto buf_gammas = gammas.request();
    auto buf_cvals = c_vals.request();
    auto buf_qs = active_qubits.request();

    ssize_t dim = buf_rho0.shape[0];
    ssize_t n_qubits = std::log2(dim);
    ssize_t p = buf_betas.shape[0];
    ssize_t q_count = buf_qs.shape[0];

    // Create a new array for the output, initialized with the input state.
    // This is the working copy that we will modify in-place.
    py::array_t<c64> rho_working_py = py::array_t<c64>(buf_rho0.shape);
    auto buf_rho = rho_working_py.request();
    c64* R = static_cast<c64*>(buf_rho.ptr);
    const c64* R0 = static_cast<const c64*>(buf_rho0.ptr);
    std::memcpy(R, R0, buf_rho0.size * sizeof(c64));

    const double* B = static_cast<const double*>(buf_betas.ptr);
    const double* G = static_cast<const double*>(buf_gammas.ptr);
    const double* C = static_cast<const double*>(buf_cvals.ptr);
    const int* Qs = static_cast<const int*>(buf_qs.ptr);

    std::vector<c64> phases(dim);

    // 2. --- Main Loop Over QAOA Layers ---
    for (ssize_t l = 0; l < p; ++l) {
        // --- Phase Application (in-place) ---
        double gamma = G[l];
        for (ssize_t i = 0; i < dim; ++i) {
            phases[i] = std::exp(c64(0, -gamma * C[i]));
        }
        for (ssize_t i = 0; i < dim; ++i) {
            for (ssize_t j = 0; j < dim; ++j) {
                R[i * dim + j] *= phases[i] * std::conj(phases[j]);
            }
        }

        // --- Mixer Application (in-place) ---
        double beta = B[l];
        double c = std::cos(beta);
        double s = std::sin(beta);
        c64 U00(c, 0), U01(0, -s), U10(0, -s), U11(c, 0);
        c64 V00(c, 0), V01(0, s), V10(0, s), V11(c, 0);

        for (ssize_t q_idx = 0; q_idx < q_count; ++q_idx) {
            int k = Qs[q_idx];
            ssize_t bit = 1LL << k;
            ssize_t grid_dim = dim / 2;

            // Efficient tile-based update
            for (ssize_t i_half = 0; i_half < grid_dim; ++i_half) {
                ssize_t i0 = (i_half & (bit - 1)) | ((i_half & ~(bit - 1)) << 1);
                ssize_t i1 = i0 | bit;

                for (ssize_t j_half = 0; j_half < grid_dim; ++j_half) {
                    ssize_t j0 = (j_half & (bit - 1)) | ((j_half & ~(bit - 1)) << 1);
                    ssize_t j1 = j0 | bit;

                    c64 r00 = R[i0*dim+j0], r01 = R[i0*dim+j1];
                    c64 r10 = R[i1*dim+j0], r11 = R[i1*dim+j1];

                    c64 t00 = U00*r00 + U01*r10, t01 = U00*r01 + U01*r11;
                    c64 t10 = U10*r00 + U11*r10, t11 = U10*r01 + U11*r11;

                    R[i0*dim+j0] = t00*V00 + t01*V10; R[i0*dim+j1] = t00*V01 + t01*V11;
                    R[i1*dim+j0] = t10*V00 + t11*V10; R[i1*dim+j1] = t10*V01 + t11*V11;
                }
            }
        }
    }

    return rho_working_py;
}

#include <complex> // Ensure this is included at the top
using c32 = std::complex<float>;

py::array_t<c32> evolve_all_layers_cpu_f32(
    py::array_t<c32> rho0,
    py::array_t<float> betas,
    py::array_t<float> gammas,
    py::array_t<float> c_vals,
    py::array_t<int> active_qubits)
{
    auto buf_rho0 = rho0.request();
    auto buf_betas = betas.request();
    auto buf_gammas = gammas.request();
    auto buf_cvals = c_vals.request();
    auto buf_qs = active_qubits.request();

    ssize_t dim = buf_rho0.shape[0];
    ssize_t n_qubits = std::log2(dim);
    ssize_t p = buf_betas.shape[0];
    ssize_t q_count = buf_qs.shape[0];

    py::array_t<c32> rho_working_py = py::array_t<c32>(buf_rho0.shape);
    auto buf_rho = rho_working_py.request();
    c32* R = static_cast<c32*>(buf_rho.ptr);
    const c32* R0 = static_cast<const c32*>(buf_rho0.ptr);
    std::memcpy(R, R0, buf_rho0.size * sizeof(c32));

    const float* B = static_cast<const float*>(buf_betas.ptr);
    const float* G = static_cast<const float*>(buf_gammas.ptr);
    const float* C = static_cast<const float*>(buf_cvals.ptr);
    const int* Qs = static_cast<const int*>(buf_qs.ptr);

    std::vector<c32> phases(dim);

    for (ssize_t l = 0; l < p; ++l) {
        float gamma = G[l];
        for (ssize_t i = 0; i < dim; ++i) {
            phases[i] = std::exp(c32(0, -gamma * C[i]));
        }
        for (ssize_t i = 0; i < dim; ++i) {
            for (ssize_t j = 0; j < dim; ++j) {
                R[i * dim + j] *= phases[i] * std::conj(phases[j]);
            }
        }

        float beta = B[l];
        float c = std::cos(beta);
        float s = std::sin(beta);
        c32 U00(c, 0), U01(0, -s), U10(0, -s), U11(c, 0);
        c32 V00(c, 0), V01(0, s), V10(0, s), V11(c, 0);

        for (ssize_t q_idx = 0; q_idx < q_count; ++q_idx) {
            int k = Qs[q_idx];
            ssize_t bit = 1LL << k;
            ssize_t grid_dim = dim / 2;

            for (ssize_t i_half = 0; i_half < grid_dim; ++i_half) {
                ssize_t i0 = (i_half & (bit - 1)) | ((i_half & ~(bit - 1)) << 1);
                ssize_t i1 = i0 | bit;
                for (ssize_t j_half = 0; j_half < grid_dim; ++j_half) {
                    ssize_t j0 = (j_half & (bit - 1)) | ((j_half & ~(bit - 1)) << 1);
                    ssize_t j1 = j0 | bit;

                    c32 r00 = R[i0*dim+j0], r01 = R[i0*dim+j1];
                    c32 r10 = R[i1*dim+j0], r11 = R[i1*dim+j1];

                    c32 t00 = U00*r00 + U01*r10, t01 = U00*r01 + U01*r11;
                    c32 t10 = U10*r00 + U11*r10, t11 = U10*r01 + U11*r11;

                    R[i0*dim+j0] = t00*V00 + t01*V10; R[i0*dim+j1] = t00*V01 + t01*V11;
                    R[i1*dim+j0] = t10*V00 + t11*V10; R[i1*dim+j1] = t10*V01 + t11*V11;
                }
            }
        }
    }
    return rho_working_py;
}