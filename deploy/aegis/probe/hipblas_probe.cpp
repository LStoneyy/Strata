// Gate A probe (PLAN.md, Phase 1): does hipBLAS/rocBLAS compute on this card, and does pinned host memory count
// against the amdgpu GTT pool?
//
//   tn             : Strata-shaped TN GemmEx (F16 and BF16 in, F32 out) at prompt sizes; reports every failure
//   cross <h> <r>  : hipBLAS handle created on device h, GEMM run on device r (mixed-arch handle reuse)
//   gemm           : device info + hipblasHgemm and hipblasGemmEx (F16 in, F32 out/compute), checked against the CPU
//   pin  <GiB>     : hipHostMalloc <GiB>, touch every page, hold 10 s (sample mem_info_gtt_used meanwhile)
//   reg  <GiB>     : malloc + hipHostRegister <GiB>, touch, hold 10 s
//
// Run one card per process (rocBLAS may abort the process on a card it has no kernels for):
//   HIP_VISIBLE_DEVICES=0 ./hipblas_probe gemm
//
// Build: hipcc -O2 --offload-arch=gfx1012 --offload-arch=gfx1200 hipblas_probe.cpp -lhipblas -o hipblas_probe
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <hipblas/hipblas.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>
#include <vector>

#define HIP_OK(x)                                                                              \
    do {                                                                                       \
        hipError_t e_ = (x);                                                                   \
        if (e_ != hipSuccess) {                                                                \
            std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(2);                                                                      \
        }                                                                                      \
    } while (0)
#define BLAS_OK(x)                                                                             \
    do {                                                                                       \
        hipblasStatus_t s_ = (x);                                                              \
        if (s_ != HIPBLAS_STATUS_SUCCESS) {                                                    \
            std::fprintf(stderr, "hipBLAS status %d at %s:%d\n", (int) s_, __FILE__, __LINE__); \
            std::exit(3);                                                                      \
        }                                                                                      \
    } while (0)

static void info() {
    int n = 0;
    HIP_OK(hipGetDeviceCount(&n));
    for (int i = 0; i < n; ++i) {
        hipDeviceProp_t p;
        HIP_OK(hipGetDeviceProperties(&p, i));
        size_t fr = 0, tot = 0;
        HIP_OK(hipSetDevice(i));
        HIP_OK(hipMemGetInfo(&fr, &tot));
        std::printf("device %d: %s  arch=%s  pci=%02x:%02x.%x  vram=%zu MiB free=%zu MiB  warp=%d\n", i, p.name,
                    p.gcnArchName, p.pciBusID, p.pciDeviceID, p.pciDomainID, tot >> 20, fr >> 20, p.warpSize);
    }
}

// Column-major C[MxN] = A[MxK] * B[KxN]
static double check(const std::vector<float>& a, const std::vector<float>& b, const std::vector<float>& c, int M,
                    int N, int K) {
    double worst = 0;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i) {
            double ref = 0;
            for (int k = 0; k < K; ++k) ref += (double) a[i + (size_t) k * M] * b[k + (size_t) j * K];
            double err = std::fabs(ref - c[i + (size_t) j * M]) / (std::fabs(ref) + 1.0);
            if (err > worst) worst = err;
        }
    return worst;
}

static int gemm() {
    info();
    HIP_OK(hipSetDevice(0));
    const int M = 384, N = 256, K = 512;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    std::vector<float> a((size_t) M * K), b((size_t) K * N);
    for (auto& v : a) v = (float) __half2float(__float2half(d(rng)));
    for (auto& v : b) v = (float) __half2float(__float2half(d(rng)));
    std::vector<__half> ah(a.size()), bh(b.size());
    for (size_t i = 0; i < a.size(); ++i) ah[i] = __float2half(a[i]);
    for (size_t i = 0; i < b.size(); ++i) bh[i] = __float2half(b[i]);

    __half *dA, *dB, *dCh;
    float* dCf;
    HIP_OK(hipMalloc(&dA, ah.size() * sizeof(__half)));
    HIP_OK(hipMalloc(&dB, bh.size() * sizeof(__half)));
    HIP_OK(hipMalloc(&dCh, (size_t) M * N * sizeof(__half)));
    HIP_OK(hipMalloc(&dCf, (size_t) M * N * sizeof(float)));
    HIP_OK(hipMemcpy(dA, ah.data(), ah.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_OK(hipMemcpy(dB, bh.data(), bh.size() * sizeof(__half), hipMemcpyHostToDevice));

    hipblasHandle_t h;
    std::printf("hipblasCreate ...\n");
    std::fflush(stdout);
    BLAS_OK(hipblasCreate(&h));

    // 1) GemmEx: F16 in, F32 out, F32 compute (Strata's prompt path shape)
    {
        const float alpha = 1.f, beta = 0.f;
        std::printf("hipblasGemmEx f16->f32 ...\n");
        std::fflush(stdout);
        BLAS_OK(hipblasGemmEx(h, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &alpha, dA, HIP_R_16F, M, dB, HIP_R_16F, K, &beta,
                              dCf, HIP_R_32F, M, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT));
        HIP_OK(hipDeviceSynchronize());
        std::vector<float> c((size_t) M * N);
        HIP_OK(hipMemcpy(c.data(), dCf, c.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::printf("GemmEx f16->f32: max rel err %.3e\n", check(a, b, c, M, N, K));
    }
    // 2) Hgemm: F16 in/out
    {
        // hipblasHalf is uint16_t unless HIPBLAS_USE_HIP_HALF: pass the bit patterns, not the converted values
        const __half one = __float2half(1.f), zero = __float2half(0.f);
        hipblasHalf alpha, beta;
        std::memcpy(&alpha, &one, sizeof alpha);
        std::memcpy(&beta, &zero, sizeof beta);
        std::printf("hipblasHgemm ...\n");
        std::fflush(stdout);
        BLAS_OK(hipblasHgemm(h, HIPBLAS_OP_N, HIPBLAS_OP_N, M, N, K, &alpha, (const hipblasHalf*) dA, M,
                             (const hipblasHalf*) dB, K, &beta, (hipblasHalf*) dCh, M));
        HIP_OK(hipDeviceSynchronize());
        std::vector<__half> ch((size_t) M * N);
        HIP_OK(hipMemcpy(ch.data(), dCh, ch.size() * sizeof(__half), hipMemcpyDeviceToHost));
        std::vector<float> c(ch.size());
        for (size_t i = 0; i < ch.size(); ++i) c[i] = __half2float(ch[i]);
        std::printf("Hgemm f16: max rel err %.3e\n", check(a, b, c, M, N, K));
    }
    BLAS_OK(hipblasDestroy(h));
    std::printf("GEMM OK\n");
    return 0;
}

// Strata's prompt GEMMs: Y[T x N] = X[T x K] * W^T, column-major as op(A)=T on the weight: C(NxT) = W(KxN)^T * X(KxT)
static int tn(hipDataType ab, const char* name, int N, int K, int T) {
    const size_t esz = 2;
    void *dW, *dX;
    float* dC;
    HIP_OK(hipMalloc(&dW, (size_t) K * N * esz));
    HIP_OK(hipMalloc(&dX, (size_t) K * T * esz));
    HIP_OK(hipMalloc(&dC, (size_t) N * T * sizeof(float)));
    HIP_OK(hipMemset(dW, 0, (size_t) K * N * esz));
    HIP_OK(hipMemset(dX, 0, (size_t) K * T * esz));
    hipblasHandle_t h;
    BLAS_OK(hipblasCreate(&h));
    const float alpha = 1.f, beta = 0.f;
    hipblasStatus_t st = hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &alpha, dW, ab, K, dX, ab, K, &beta, dC,
                                       HIP_R_32F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    hipError_t e = hipDeviceSynchronize();
    hipError_t le = hipGetLastError();
    std::printf("TN %s N=%d K=%d T=%d: status %d, sync %s, last %s\n", name, N, K, T, (int) st, hipGetErrorString(e),
                hipGetErrorString(le));
    hipblasDestroy(h);
    hipFree(dW); hipFree(dX); hipFree(dC);
    return st == HIPBLAS_STATUS_SUCCESS && e == hipSuccess && le == hipSuccess ? 0 : 1;
}

static int tnall() {
    info();
    HIP_OK(hipSetDevice(0));
    int bad = 0;
    const int shapes[][3] = {{2560, 2560, 64}, {12288, 2560, 2048}, {2560, 6144, 2048}, {512, 2560, 2048},
                             {2560, 10240, 512}, {640, 2560, 8192}, {248320, 2560, 16}};
    for (auto& s : shapes) {
        bad += tn(HIP_R_16F, "f16", s[0], s[1], s[2]);
        bad += tn(HIP_R_16BF, "bf16", s[0], s[1], s[2]);
    }
    std::printf(bad ? "TN FAILURES: %d\n" : "TN OK\n", bad);
    return bad ? 5 : 0;
}

// A handle made on device <hd>, used for a GEMM whose buffers and stream live on device <rd> (the mixed-arch hypothesis)
static int cross(int hd, int rd) {
    info();
    HIP_OK(hipSetDevice(hd));
    hipblasHandle_t h;
    BLAS_OK(hipblasCreate(&h));
    HIP_OK(hipSetDevice(rd));
    hipStream_t st;
    HIP_OK(hipStreamCreate(&st));
    BLAS_OK(hipblasSetStream(h, st));
    const int N = 2560, K = 2560, T = 2048;
    void *dW, *dX; float* dC;
    HIP_OK(hipMalloc(&dW, (size_t) K * N * 2)); HIP_OK(hipMalloc(&dX, (size_t) K * T * 2)); HIP_OK(hipMalloc(&dC, (size_t) N * T * 4));
    const float alpha = 1.f, beta = 0.f;
    hipblasStatus_t s = hipblasGemmEx(h, HIPBLAS_OP_T, HIPBLAS_OP_N, N, T, K, &alpha, dW, HIP_R_16F, K, dX, HIP_R_16F, K, &beta,
                                      dC, HIP_R_32F, N, HIPBLAS_COMPUTE_32F, HIPBLAS_GEMM_DEFAULT);
    hipError_t e = hipStreamSynchronize(st), le = hipGetLastError();
    std::printf("handle on %d, run on %d: status %d, sync %s, last %s\n", hd, rd, (int) s, hipGetErrorString(e), hipGetErrorString(le));
    return 0;
}

static int pin(double gib, bool reg) {
    info();
    HIP_OK(hipSetDevice(0));
    const size_t bytes = (size_t) (gib * (1ull << 30));
    void* p = nullptr;
    if (reg) {
        p = std::aligned_alloc(4096, bytes);
        if (!p) { std::fprintf(stderr, "aligned_alloc failed\n"); return 4; }
        std::memset(p, 1, bytes);
        HIP_OK(hipHostRegister(p, bytes, hipHostRegisterDefault));
    } else {
        HIP_OK(hipHostMalloc(&p, bytes, hipHostMallocDefault));
        std::memset(p, 1, bytes);
    }
    std::printf("PINNED %.2f GiB via %s, holding 10 s\n", gib, reg ? "hipHostRegister" : "hipHostMalloc");
    std::fflush(stdout);
    std::this_thread::sleep_for(std::chrono::seconds(10));
    if (reg) { HIP_OK(hipHostUnregister(p)); std::free(p); } else HIP_OK(hipHostFree(p));
    std::printf("released\n");
    return 0;
}

int main(int argc, char** argv) {
    if (argc >= 2 && !std::strcmp(argv[1], "gemm")) return gemm();
    if (argc >= 2 && !std::strcmp(argv[1], "tn")) return tnall();
    if (argc >= 4 && !std::strcmp(argv[1], "cross")) return cross(std::atoi(argv[2]), std::atoi(argv[3]));
    if (argc >= 3 && !std::strcmp(argv[1], "pin")) return pin(std::atof(argv[2]), false);
    if (argc >= 3 && !std::strcmp(argv[1], "reg")) return pin(std::atof(argv[2]), true);
    std::fprintf(stderr, "usage: %s gemm | pin <GiB> | reg <GiB>\n", argv[0]);
    return 1;
}
