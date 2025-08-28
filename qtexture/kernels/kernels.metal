#include <metal_stdlib>
using namespace metal;

struct ParamsF32 {
    uint dim, bit, period, pad;
    float2 U00, U01, U10, U11;
    float2 V00, V01, V10, V11;
};

inline float2 cmul(float2 a, float2 b) {
    return float2(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

kernel void apply_phase_f32(device float2* rho      [[buffer(0)]],
                            device const float2* ph [[buffer(1)]],
                            constant uint& dim      [[buffer(2)]],
                            uint2 gid               [[thread_position_in_grid]]) {
    uint i = gid.x;
    uint j = gid.y;
    if (i >= dim || j >= dim) return;

    float2 pi = ph[i];
    float2 pj = ph[j];
    float2 pj_conj = float2(pj.x, -pj.y);

    uint idx = i * dim + j;
    rho[idx] = cmul(cmul(pi, rho[idx]), pj_conj);
}

kernel void apply_unitary_tile_f32(device float2* rho [[buffer(0)]],
                                   constant ParamsF32& p [[buffer(1)]],
                                   uint2 gid [[thread_position_in_grid]]) {
    uint grid_dim = p.dim / 2;
    if (gid.x >= grid_dim || gid.y >= grid_dim) return;

    uint i0 = (gid.x & (p.bit - 1)) | ((gid.x & ~(p.bit - 1)) << 1);
    uint i1 = i0 | p.bit;
    uint j0 = (gid.y & (p.bit - 1)) | ((gid.y & ~(p.bit - 1)) << 1);
    uint j1 = j0 | p.bit;

    float2 A00 = rho[i0 * p.dim + j0]; float2 A01 = rho[i0 * p.dim + j1];
    float2 A10 = rho[i1 * p.dim + j0]; float2 A11 = rho[i1 * p.dim + j1];

    float2 T00 = cmul(p.U00, A00) + cmul(p.U01, A10);
    float2 T01 = cmul(p.U00, A01) + cmul(p.U01, A11);
    float2 T10 = cmul(p.U10, A00) + cmul(p.U11, A10);
    float2 T11 = cmul(p.U10, A01) + cmul(p.U11, A11);

    rho[i0 * p.dim + j0] = cmul(T00, p.V00) + cmul(T01, p.V10);
    rho[i0 * p.dim + j1] = cmul(T00, p.V01) + cmul(T01, p.V11);
    rho[i1 * p.dim + j0] = cmul(T10, p.V00) + cmul(T11, p.V10);
    rho[i1 * p.dim + j1] = cmul(T10, p.V01) + cmul(T11, p.V11);
}

kernel void qaoa_evolution_small_system(
    device float2* rho_global [[buffer(0)]],
    device const float* betas [[buffer(1)]],
    device const float* gammas [[buffer(2)]],
    device const float* c_vals [[buffer(3)]],
    device const int* qs [[buffer(4)]],
    constant uint& n [[buffer(5)]],
    constant uint& p [[buffer(6)]],
    constant uint& qcount [[buffer(7)]],
    threadgroup float2* rho_tg [[threadgroup(10)]],
    uint2 tid [[thread_position_in_threadgroup]])
{
    //if (tid.x == 0 && tid.y == 0) {
    //    rho_global[0] = float2(betas[0], 999.0f);
    //}
    //return; // Skip the rest of the kernel

    uint dim = 1 << n;
    uint i = tid.x;
    uint j = tid.y;

    rho_tg[i * dim + j] = rho_global[i * dim + j];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint l = 0; l < p; ++l) {
        float g = gammas[l];
        float angle_i = -g * c_vals[i];
        float angle_j = -g * c_vals[j];
        float2 phase_i = float2(cos(angle_i), sin(angle_i));
        float2 phase_j_conj = float2(cos(angle_j), -sin(angle_j));
        rho_tg[i * dim + j] = cmul(cmul(phase_i, rho_tg[i * dim + j]), phase_j_conj);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float b = betas[l];
        if (b != 0.0) {
            float c = cos(b);
            float s = sin(b);
            float2 U00 = float2(c, 0.0f); float2 U11 = float2(c, 0.0f);
            float2 U01 = float2(0.0f, -s); float2 U10 = float2(0.0f, -s);
            float2 V00 = float2(c, 0.0f); float2 V11 = float2(c, 0.0f);
            float2 V01 = float2(0.0f, s); float2 V10 = float2(0.0f, s);

            for (uint t = 0; t < qcount; ++t) {
                uint bit = 1 << qs[t];
                if ((i & bit) == 0 && (j & bit) == 0) {
                    uint i1 = i | bit;
                    uint j1 = j | bit;

                    float2 A00 = rho_tg[i  * dim + j ]; float2 A01 = rho_tg[i  * dim + j1];
                    float2 A10 = rho_tg[i1 * dim + j ]; float2 A11 = rho_tg[i1 * dim + j1];

                    float2 T00 = cmul(U00, A00) + cmul(U01, A10);
                    float2 T01 = cmul(U00, A01) + cmul(U01, A11);
                    float2 T10 = cmul(U10, A00) + cmul(U11, A10);
                    float2 T11 = cmul(U10, A01) + cmul(U11, A11);

                    rho_tg[i  * dim + j ] = cmul(T00, V00) + cmul(T01, V10);
                    rho_tg[i  * dim + j1] = cmul(T00, V01) + cmul(T01, V11);
                    rho_tg[i1 * dim + j ] = cmul(T10, V00) + cmul(T11, V10);
                    rho_tg[i1 * dim + j1] = cmul(T10, V01) + cmul(T11, V11);
                }
                 threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
    rho_global[i * dim + j] = rho_tg[i * dim + j];
}