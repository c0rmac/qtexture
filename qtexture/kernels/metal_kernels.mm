// metal_kernels.mm (Objective‑C++)
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <complex>
#include <vector>
#import <simd/simd.h>   // <- this is the missing header

namespace py = pybind11;
using c32 = std::complex<float>;

struct ParamsF32 {
    uint32_t dim, bit, period, pad;
    simd_float2 U00, U01, U10, U11;
    simd_float2 V00, V01, V10, V11;
};

static inline simd_float2 to_f2(const c32& z) {
    return simd_make_float2(z.real(), z.imag());
}

// Globals used by your bridge functions
id<MTLDevice>                g_device            = nil;
id<MTLCommandQueue>          g_queue             = nil;
id<MTLComputePipelineState>  g_pso_f32           = nil; // unitary‑tile kernel
id<MTLComputePipelineState>  g_pso_phase_f32     = nil; // diagonal phase kernel
// Add a new global pipeline state at the top with the others
id<MTLComputePipelineState>  g_pso_small_sys_f32 = nil;

void metal_init(const std::string& metallib_path) {
    @autoreleasepool {
        g_device = MTLCreateSystemDefaultDevice();
        if (!g_device) {
            throw std::runtime_error("Metal device not available");
        }
        g_queue = [g_device newCommandQueue];
        if (!g_queue) {
            throw std::runtime_error("Failed to create Metal command queue");
        }

        NSError *err = nil;
        NSString *nsPath = [NSString stringWithUTF8String:metallib_path.c_str()];
        id<MTLLibrary> lib = [g_device newLibraryWithFile:nsPath error:&err];
        if (!lib || err) {
            throw std::runtime_error("Failed to load metallib");
        }

        // Pipeline for the 2×2 unitary‑tile kernel
        {
            id<MTLFunction> fn = [lib newFunctionWithName:@"apply_unitary_tile_f32"];
            if (!fn) throw std::runtime_error("Kernel apply_unitary_tile_f32 not found in metallib");
            g_pso_f32 = [g_device newComputePipelineStateWithFunction:fn error:&err];
            if (!g_pso_f32 || err) throw std::runtime_error("Failed to create pipeline for unitary kernel");
        }

        // Pipeline for the diagonal cost‑phase kernel
        {
            id<MTLFunction> fn = [lib newFunctionWithName:@"apply_phase_f32"];
            if (!fn) throw std::runtime_error("Kernel apply_phase_f32 not found in metallib");
            g_pso_phase_f32 = [g_device newComputePipelineStateWithFunction:fn error:&err];
            if (!g_pso_phase_f32 || err) throw std::runtime_error("Failed to create pipeline for phase kernel");
        }

        // Pipeline for the small system specialized kernel
        {
            id<MTLFunction> fn = [lib newFunctionWithName:@"qaoa_evolution_small_system"];
            if (!fn) throw std::runtime_error("Kernel qaoa_evolution_small_system not found in metallib");
            g_pso_small_sys_f32 = [g_device newComputePipelineStateWithFunction:fn error:&err];
            if (!g_pso_small_sys_f32 || err) throw std::runtime_error("Failed to create pipeline for small system kernel");
        }
    }
}

// Rx(θ) for θ = 2*beta (complex64)
static inline void rx_matrix(float theta, c32& U00, c32& U01, c32& U10, c32& U11) {
    float c = std::cosf(theta * 0.5f);
    float s = std::sinf(theta * 0.5f);
    U00 = c32(c, 0.0f);
    U11 = c32(c, 0.0f);
    // -i s
    U01 = c32(0.0f, -s);
    U10 = c32(0.0f, -s);
}

void apply_layers_metal_f32(py::array_t<c32> rho,
                            py::array_t<float> betas,
                            py::array_t<float> gammas,
                            py::array_t<double> c_vals,
                            py::array qs,
                            int n)
{
    if (!g_pso_f32 || !g_pso_phase_f32) {
        throw std::runtime_error("Metal kernels not initialized");
    }

    auto br = rho.request();
    auto bb = betas.request();
    auto bg = gammas.request();
    auto bc = c_vals.request();
    auto bq = qs.request();

    // Shapes and basic checks
    if (br.ndim != 2 || br.shape[0] != br.shape[1]) throw std::runtime_error("rho must be square");
    const uint32_t dim = static_cast<uint32_t>(br.shape[0]);
    if ((1u << n) != dim) throw std::runtime_error("dim must equal 2^n");
    if (bb.ndim != 1 || bg.ndim != 1) throw std::runtime_error("betas and gammas must be 1D");
    if (bb.shape[0] != bg.shape[0]) throw std::runtime_error("betas and gammas must have same length");
    const uint32_t p = static_cast<uint32_t>(bb.shape[0]);
    if (bc.ndim != 1 || static_cast<uint32_t>(bc.shape[0]) != dim) throw std::runtime_error("c_vals length mismatch");
    if (bq.ndim != 1) throw std::runtime_error("qs must be 1D");
    if (!(br.strides[1] == (ssize_t)sizeof(c32) && br.strides[0] == (ssize_t)sizeof(c32) * dim))
        throw std::runtime_error("rho must be C-contiguous complex64");

    const float* betas_ptr  = static_cast<const float*>(bb.ptr);
    const float* gammas_ptr = static_cast<const float*>(bg.ptr);
    const double* c_ptr     = static_cast<const double*>(bc.ptr);
    const int* qs_ptr       = static_cast<const int*>(bq.ptr);
    const uint32_t qcount   = static_cast<uint32_t>(bq.shape[0]);

    py::gil_scoped_release release;

    @autoreleasepool {
        // Wrap rho (shared, writes directly into numpy memory)
        id<MTLBuffer> bufR = [g_device newBufferWithBytesNoCopy:br.ptr
                                                         length:(size_t)dim * dim * sizeof(c32)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!bufR) throw std::runtime_error("Failed to create shared buffer for rho");

        // Precompute phases for all layers into one big host buffer (p x dim)
        const size_t phases_bytes = (size_t)p * dim * sizeof(c32);
        id<MTLBuffer> bufP = [g_device newBufferWithLength:phases_bytes options:MTLResourceStorageModeShared];
        if (!bufP) throw std::runtime_error("Failed to allocate phases buffer");
        {
            c32* P = static_cast<c32*>([bufP contents]);
            for (uint32_t l = 0; l < p; ++l) {
                const float g = gammas_ptr[l];
                c32* Pl = P + (size_t)l * dim;
                for (uint32_t i = 0; i < dim; ++i) {
                    float angle = -(float)g * (float)c_ptr[i];
                    float cs = std::cosf(angle);
                    float sn = std::sinf(angle);
                    Pl[i] = c32(cs, sn);  // exp(i*angle) with angle negative -> exp(-i g c)
                }
            }
        }

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        // We reuse one encoder, switching pipeline as needed
        // 1) For each layer: apply phase with per-layer offset, then mixers over all qs
        for (uint32_t l = 0; l < p; ++l) {
            // Phase
            [enc setComputePipelineState:g_pso_phase_f32];
            [enc setBuffer:bufR offset:0 atIndex:0];
            [enc setBuffer:bufP offset:(size_t)l * dim * sizeof(c32) atIndex:1];
            [enc setBytes:&dim length:sizeof(uint32_t) atIndex:2];

            MTLSize tgsP     = MTLSizeMake(16, 16, 1);
            MTLSize tgcountP = MTLSizeMake((dim + tgsP.width  - 1) / tgsP.width,
                                           (dim + tgsP.height - 1) / tgsP.height,
                                           1);
            [enc dispatchThreadgroups:tgcountP threadsPerThreadgroup:tgsP];

            // Mixers: Rx(2*beta_l) on all active qubits
            c32 U00, U01, U10, U11;
            rx_matrix(2.0f * betas_ptr[l], U00, U01, U10, U11);
            c32 V00 = std::conj(U00), V01 = std::conj(U10), V10 = std::conj(U01), V11 = std::conj(U11);

            [enc setComputePipelineState:g_pso_f32];
            [enc setBuffer:bufR offset:0 atIndex:0];

            for (uint32_t t = 0; t < qcount; ++t) {
                int k = qs_ptr[t];
                if (k < 0 || k >= n) continue; // guard

                ParamsF32 pbuf{};
                pbuf.dim    = dim;
                pbuf.bit    = (1u << k);
                // The 'period' parameter is no longer used by the corrected kernel
                pbuf.period = 0;
                pbuf.pad    = 0;
                pbuf.U00 = to_f2(U00); pbuf.U01 = to_f2(U01);
                pbuf.U10 = to_f2(U10); pbuf.U11 = to_f2(U11);
                pbuf.V00 = to_f2(V00); pbuf.V01 = to_f2(V01);
                pbuf.V10 = to_f2(V10); pbuf.V11 = to_f2(V11);

                [enc setBytes:&pbuf length:sizeof(ParamsF32) atIndex:1];

                // --- START: REVISED DISPATCH LOGIC ---
                // The new kernel needs a grid of size (dim/2, dim/2)
                const uint32_t grid_dim = dim / 2;
                MTLSize tgsU     = MTLSizeMake(16, 16, 1);
                MTLSize tgcountU = MTLSizeMake((grid_dim + tgsU.width  - 1) / tgsU.width,
                                               (grid_dim + tgsU.height - 1) / tgsU.height,
                                               1);
                [enc dispatchThreadgroups:tgcountU threadsPerThreadgroup:tgsU];
                // --- END: REVISED DISPATCH LOGIC ---
            }
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

static inline void fill_pair(float dst[2], const c32& z) {
    dst[0] = z.real();
    dst[1] = z.imag();
}

// Fused layer: phase then mixers in one command buffer
void apply_phase_and_mixer_layer_metal_f32(py::array_t<std::complex<float>> rho,
                                           py::array_t<std::complex<float>> phases,
                                           py::array_t<std::complex<float>> U,
                                           py::array qs,
                                           int n)
{
    using c32 = std::complex<float>;
    if (!g_pso_phase_f32 || !g_pso_f32) throw std::runtime_error("Metal kernels not initialized");

    auto br = rho.request(), bp = phases.request(), bU = U.request(), bQ = qs.request();
    const uint32_t dim = static_cast<uint32_t>(br.shape[0]);

    // Validate shapes/dtypes/contiguity
    if (br.ndim != 2 || br.shape[0] != br.shape[1]) throw std::runtime_error("rho must be square");
    if (bp.ndim != 1 || static_cast<uint32_t>(bp.shape[0]) != dim) throw std::runtime_error("phases length mismatch");
    if (bU.ndim != 2 || bU.shape[0] != 2 || bU.shape[1] != 2) throw std::runtime_error("U must be 2x2");
    if (bQ.ndim != 1) throw std::runtime_error("qs must be 1D");
    if ((1u << n) != dim) throw std::runtime_error("dim must equal 2^n");

    auto need = [&](ssize_t stride, ssize_t expect) {
        return stride != expect;
    };

    if (need(br.strides[1], sizeof(c32)) || need(br.strides[0], sizeof(c32) * dim))
        throw std::runtime_error("rho must be C-contiguous complex64");
    if (need(bp.strides[0], sizeof(c32)))
        throw std::runtime_error("phases must be contiguous complex64");
    if (need(bU.strides[1], sizeof(c32)) || need(bU.strides[0], sizeof(c32) * 2))
        throw std::runtime_error("U must be C-contiguous complex64");

    // Prepack U and V = U†
    const auto* UU = static_cast<const c32*>(bU.ptr);
    const c32 U00 = UU[0], U01 = UU[1], U10 = UU[2], U11 = UU[3];
    const c32 V00 = std::conj(UU[0]), V01 = std::conj(UU[2]),
               V10 = std::conj(UU[1]), V11 = std::conj(UU[3]);

    py::gil_scoped_release release;

    @autoreleasepool {
        id<MTLBuffer> bufR = [g_device newBufferWithBytesNoCopy:br.ptr
                                                         length:(size_t)dim * dim * sizeof(c32)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        id<MTLBuffer> bufP = [g_device newBufferWithBytesNoCopy:bp.ptr
                                                         length:(size_t)dim * sizeof(c32)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!bufR || !bufP) throw std::runtime_error("Failed to create shared buffers");

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];

        // 1) Phase
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_pso_phase_f32];
            [enc setBuffer:bufR offset:0 atIndex:0];
            [enc setBuffer:bufP offset:0 atIndex:1];
            [enc setBytes:&(dim) length:sizeof(uint32_t) atIndex:2];

            MTLSize tgs     = MTLSizeMake(16, 16, 1);
            MTLSize tgcount = MTLSizeMake((dim + tgs.width  - 1) / tgs.width,
                                          (dim + tgs.height - 1) / tgs.height,
                                          1);
            [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgs];
            [enc endEncoding];
        }

        // 2) Mixers (reuse the same rho buffer)
        {
            id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
            [enc setComputePipelineState:g_pso_f32];
            [enc setBuffer:bufR offset:0 atIndex:0];

            // aligned params struct
            struct ParamsF32 {
                uint32_t dim, bit, period, pad;
                simd_float2 U00, U01, U10, U11;
                simd_float2 V00, V01, V10, V11;
            };
            auto to_f2 = [](const c32& z){ return simd_make_float2(z.real(), z.imag()); };

            const int* qs_ptr = static_cast<const int*>(bQ.ptr);
            for (ssize_t t = 0; t < bQ.shape[0]; ++t) {
                int k = qs_ptr[t];
                if (k < 0 || k >= n) throw std::runtime_error("qubit index out of range");

                ParamsF32 p{};
                p.dim = dim; p.bit = (1u << k); p.period = (p.bit << 1); p.pad = 0;
                p.U00 = to_f2(U00); p.U01 = to_f2(U01); p.U10 = to_f2(U10); p.U11 = to_f2(U11);
                p.V00 = to_f2(V00); p.V01 = to_f2(V01); p.V10 = to_f2(V10); p.V11 = to_f2(V11);

                [enc setBytes:&p length:sizeof(ParamsF32) atIndex:1];

                const uint32_t tiles = dim / p.period;
                MTLSize tgs     = MTLSizeMake(16, 8, 1);
                MTLSize tgcount = MTLSizeMake((tiles + tgs.width  - 1) / tgs.width,
                                              (tiles + tgs.height - 1) / tgs.height,
                                              1);
                [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgs];
            }
            [enc endEncoding];
        }

        [cmd commit];
        [cmd waitUntilCompleted];
    }
}


// Single-gate GPU path (kept for compatibility)
void apply_1q_unitary_density_metal_f32(py::array_t<std::complex<float>> rho,
                                        py::array_t<std::complex<float>> U,
                                        int k, int n)
{
    if (!g_pso_f32) throw std::runtime_error("Metal not initialized. Call metal_init() first.");

    using c32 = std::complex<float>;
    if (!py::isinstance<py::array_t<c32>>(rho) || !py::isinstance<py::array_t<c32>>(U)) {
        throw std::runtime_error("Metal f32 path expects complex64 arrays");
    }

    auto br = rho.request();
    auto bU = U.request();

    if (br.ndim != 2) throw std::runtime_error("rho must be 2D");
    if (br.shape[0] != br.shape[1]) throw std::runtime_error("rho must be square");
    if (bU.ndim != 2 || bU.shape[0] != 2 || bU.shape[1] != 2) throw std::runtime_error("U must be 2x2");
    if (k < 0 || k >= n) throw std::runtime_error("k out of range");

    const uint32_t dim = static_cast<uint32_t>(br.shape[0]);
    if ((1u << n) != dim) throw std::runtime_error("dim must equal 2^n");

    // Contiguity checks
    if (!(br.strides[1] == static_cast<ssize_t>(sizeof(c32)) &&
          br.strides[0] == static_cast<ssize_t>(sizeof(c32)) * dim)) {
        throw std::runtime_error("rho must be C-contiguous complex64");
    }
    if (!(bU.strides[1] == static_cast<ssize_t>(sizeof(c32)) &&
          bU.strides[0] == static_cast<ssize_t>(sizeof(c32)) * 2)) {
        throw std::runtime_error("U must be C-contiguous complex64");
    }

    // Pack params
    const auto* UU = static_cast<const c32*>(bU.ptr);
    const c32 U00 = UU[0], U01 = UU[1], U10 = UU[2], U11 = UU[3];
    const c32 V00 = std::conj(UU[0]), V01 = std::conj(UU[2]),
               V10 = std::conj(UU[1]), V11 = std::conj(UU[3]);

    ParamsF32 p{};
    p.dim    = dim;
    p.bit    = (1u << k);
    p.period = (p.bit << 1);
    p.pad    = 0;

    p.U00 = to_f2(U00); p.U01 = to_f2(U01); p.U10 = to_f2(U10); p.U11 = to_f2(U11);
    p.V00 = to_f2(V00); p.V01 = to_f2(V01); p.V10 = to_f2(V10); p.V11 = to_f2(V11);

    py::gil_scoped_release release;

    @autoreleasepool {
        // Wrap rho once
        const size_t bytes_rho = static_cast<size_t>(dim) * dim * sizeof(c32);
        id<MTLBuffer> bufR = [g_device newBufferWithBytesNoCopy:br.ptr
                                                         length:bytes_rho
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!bufR) throw std::runtime_error("Failed to create shared buffer for rho");

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pso_f32];
        [enc setBuffer:bufR offset:0 atIndex:0];
        [enc setBytes:&p length:sizeof(ParamsF32) atIndex:1];

        const uint32_t tiles = dim / p.period;
        MTLSize tgs     = MTLSizeMake(16, 8, 1);
        MTLSize tgcount = MTLSizeMake((tiles + tgs.width  - 1) / tgs.width,
                                      (tiles + tgs.height - 1) / tgs.height,
                                      1);

        [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}


// Batched (layer) GPU path: apply same 2x2 U to multiple target qubits in one command buffer.
// Apply the same 2x2 unitary to multiple target qubits in one GPU command buffer (float2/complex64 path)

void apply_1q_unitary_density_layer_metal_f32(py::array_t<std::complex<float>> rho,
                                              py::array_t<std::complex<float>> U,
                                              py::array qs,
                                              int n)
{
    if (!g_pso_f32) throw std::runtime_error("Metal not initialized. Call metal_init() first.");

    if (!py::isinstance<py::array_t<c32>>(rho) || !py::isinstance<py::array_t<c32>>(U)) {
        throw std::runtime_error("Metal f32 path expects complex64 arrays");
    }
    auto br = rho.request();
    auto bU = U.request();
    auto bQ = qs.request();

    if (br.ndim != 2) throw std::runtime_error("rho must be 2D");
    if (br.shape[0] != br.shape[1]) throw std::runtime_error("rho must be square");
    if (bU.ndim != 2 || bU.shape[0] != 2 || bU.shape[1] != 2) throw std::runtime_error("U must be 2x2");
    if (bQ.ndim != 1) throw std::runtime_error("qs must be 1D");

    const uint32_t dim = static_cast<uint32_t>(br.shape[0]);
    if ((1u << n) != dim) throw std::runtime_error("dim must equal 2^n");

    // C-contiguity checks for complex64
    if (!(br.strides[1] == static_cast<ssize_t>(sizeof(c32)) &&
          br.strides[0] == static_cast<ssize_t>(sizeof(c32)) * dim)) {
        throw std::runtime_error("rho must be C-contiguous complex64");
    }
    if (!(bU.strides[1] == static_cast<ssize_t>(sizeof(c32)) &&
          bU.strides[0] == static_cast<ssize_t>(sizeof(c32)) * 2)) {
        throw std::runtime_error("U must be C-contiguous complex64");
    }

    // Prepack U and V = U† as simd_float2
    const auto* UU = static_cast<const c32*>(bU.ptr);
    const c32 U00 = UU[0], U01 = UU[1], U10 = UU[2], U11 = UU[3];
    const c32 V00 = std::conj(UU[0]), V01 = std::conj(UU[2]),
               V10 = std::conj(UU[1]), V11 = std::conj(UU[3]);

    py::gil_scoped_release release;

    @autoreleasepool {
        // Wrap rho once (unified memory)
        const size_t bytes_rho = static_cast<size_t>(dim) * dim * sizeof(c32);
        id<MTLBuffer> bufR = [g_device newBufferWithBytesNoCopy:br.ptr
                                                         length:bytes_rho
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!bufR) throw std::runtime_error("Failed to create shared buffer for rho");

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pso_f32];
        [enc setBuffer:bufR offset:0 atIndex:0];

        // Encode one dispatch per target qubit, but all within this single encoder/cmd buffer
        const int* qs_ptr = static_cast<const int*>(bQ.ptr);
        for (ssize_t t = 0; t < bQ.shape[0]; ++t) {
            const int k = qs_ptr[t];
            if (k < 0 || k >= n) throw std::runtime_error("qubit index out of range");

            ParamsF32 p{};
            p.dim    = dim;
            p.bit    = (1u << k);
            p.period = (p.bit << 1);
            p.pad    = 0;

            p.U00 = to_f2(U00); p.U01 = to_f2(U01);
            p.U10 = to_f2(U10); p.U11 = to_f2(U11);
            p.V00 = to_f2(V00); p.V01 = to_f2(V01);
            p.V10 = to_f2(V10); p.V11 = to_f2(V11);

            [enc setBytes:&p length:sizeof(ParamsF32) atIndex:1];

            const uint32_t tiles = dim / p.period;
            MTLSize tgs     = MTLSizeMake(16, 8, 1); // tune: 16x8, 8x16, 32x4
            MTLSize tgcount = MTLSizeMake((tiles + tgs.width  - 1) / tgs.width,
                                          (tiles + tgs.height - 1) / tgs.height,
                                          1);

            [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgs];
        }

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted]; // you can make this optional to overlap layers
    }
}


void apply_cost_phase_inplace_metal_f32(py::array_t<c32> rho,
                                        py::array_t<c32> phases)
{
    if (!g_pso_phase_f32) throw std::runtime_error("Metal phase kernel not initialized");

    auto br = rho.request();
    auto bp = phases.request();

    if (br.ndim != 2 || br.shape[0] != br.shape[1])
        throw std::runtime_error("rho must be square");
    const uint32_t dim = static_cast<uint32_t>(br.shape[0]);
    if (bp.ndim != 1 || static_cast<uint32_t>(bp.shape[0]) != dim)
        throw std::runtime_error("phases must be length dim");

    // Contiguity and dtype checks (complex64)
    if (!(br.strides[1] == (ssize_t)sizeof(c32) &&
          br.strides[0] == (ssize_t)sizeof(c32) * dim))
        throw std::runtime_error("rho must be C-contiguous complex64");
    if (!(bp.strides[0] == (ssize_t)sizeof(c32)))
        throw std::runtime_error("phases must be contiguous complex64");

    py::gil_scoped_release release;

    @autoreleasepool {
        // Wrap unified memory
        id<MTLBuffer> bufR = [g_device newBufferWithBytesNoCopy:br.ptr
                                                         length:(size_t)dim * dim * sizeof(c32)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        id<MTLBuffer> bufP = [g_device newBufferWithBytesNoCopy:bp.ptr
                                                         length:(size_t)dim * sizeof(c32)
                                                        options:MTLResourceStorageModeShared
                                                    deallocator:nil];
        if (!bufR || !bufP) throw std::runtime_error("Failed to create shared buffers");

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
        [enc setComputePipelineState:g_pso_phase_f32];
        [enc setBuffer:bufR offset:0 atIndex:0];
        [enc setBuffer:bufP offset:0 atIndex:1];
        [enc setBytes:&(dim) length:sizeof(uint32_t) atIndex:2];

        MTLSize tgs     = MTLSizeMake(16, 16, 1);
        MTLSize tgcount = MTLSizeMake((dim + tgs.width  - 1) / tgs.width,
                                      (dim + tgs.height - 1) / tgs.height,
                                      1);

        [enc dispatchThreadgroups:tgcount threadsPerThreadgroup:tgs];
        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];
    }
}

void apply_layers_small_system_metal_f32(py::array_t<c32> rho,
                                        py::array_t<float> betas,
                                        py::array_t<float> gammas,
                                        py::array_t<float> c_vals,
                                        py::array qs,
                                        int n)
{
    if (!g_pso_small_sys_f32) throw std::runtime_error("Metal small system kernel not initialized");

    auto br = rho.request(), bb = betas.request(), bg = gammas.request(), bc = c_vals.request(), bq = qs.request();
    const uint32_t dim = 1 << n;
    const uint32_t p = bb.shape[0];
    const uint32_t qcount = bq.shape[0];

    py::gil_scoped_release release;

    @autoreleasepool {
        // --- START: CORRECTED MEMORY HANDLING ---
        // 1. Create a new Metal buffer and COPY the rho data into it.
        //    This is more robust than the zero-copy version.
        id<MTLBuffer> bufR = [g_device newBufferWithBytes:br.ptr length:((size_t)br.size*br.itemsize) options:MTLResourceStorageModeShared];
        // --- END: CORRECTED MEMORY HANDLING ---

        id<MTLBuffer> bufB = [g_device newBufferWithBytes:bb.ptr length:((size_t)bb.size*bb.itemsize) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufG = [g_device newBufferWithBytes:bg.ptr length:((size_t)bg.size*bg.itemsize) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufC = [g_device newBufferWithBytes:bc.ptr length:((size_t)bc.size*bc.itemsize) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufQ = [g_device newBufferWithBytes:bq.ptr length:((size_t)bq.size*bq.itemsize) options:MTLResourceStorageModeShared];

        id<MTLCommandBuffer> cmd = [g_queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];

        [enc setComputePipelineState:g_pso_small_sys_f32];
        [enc setBuffer:bufR offset:0 atIndex:0];
        [enc setBuffer:bufB offset:0 atIndex:1];
        [enc setBuffer:bufG offset:0 atIndex:2];
        [enc setBuffer:bufC offset:0 atIndex:3];
        [enc setBuffer:bufQ offset:0 atIndex:4];
        [enc setBytes:&n length:sizeof(uint32_t) atIndex:5];
        [enc setBytes:&p length:sizeof(uint32_t) atIndex:6];
        [enc setBytes:&qcount length:sizeof(uint32_t) atIndex:7];

        NSUInteger tg_mem_size = dim * dim * sizeof(c32);
        [enc setThreadgroupMemoryLength:tg_mem_size atIndex:10];

        MTLSize tgSize = MTLSizeMake(dim, dim, 1);
        if (tgSize.width > g_pso_small_sys_f32.maxTotalThreadsPerThreadgroup) {
             throw std::runtime_error("System size is too large for this specialized kernel's threadgroup.");
        }
        [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:tgSize];

        [enc endEncoding];
        [cmd commit];
        [cmd waitUntilCompleted];

        // --- START: CORRECTED MEMORY HANDLING ---
        // 2. After the kernel is done, copy the results from the Metal buffer
        //    back to the original NumPy array's memory.
        void* result_ptr = [bufR contents];
        std::memcpy(br.ptr, result_ptr, (size_t)br.size * br.itemsize);
        // --- END: CORRECTED MEMORY HANDLING ---
    }
}