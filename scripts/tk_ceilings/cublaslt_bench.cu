// cuBLASLt GEMM ceiling: c[m,n] = a[m,k] · b[n,k]ᵀ, 16-bit in and out, f32 compute.
// Row-major operands as tk's GEMM takes them, expressed column-major for cuBLASLt:
// cᵀ[n,m] = b[n,k] · aᵀ  ->  op(A)=T on b (k×n col-major, ld k), op(B)=N on a (k×m, ld k), C n×m ld n.
// Every heuristic algorithm (up to 64) is timed in 3 alternating rounds; per shape it prints the
// default (the heuristic's first) and the best:
//   ref|cublaslt|dtype|MxNxK|algo|ns|TFLOP/s|sustained_ns|sustained_TFLOP/s
// ns: min over rounds of the min over 30 single calls, each bracketed by cudaEvents on the stream after warmup to a plateau.
// sustained: mean of 200 back-to-back calls between two events. TFLOP/s = 2·M·N·K / ns / 1e3.
// Build: nvcc -O3 -std=c++17 -arch=sm_90a cublaslt_bench.cu -lcublasLt -o cublaslt_bench
// Usage: cublaslt_bench [bf16|f16] [MxNxK ...]
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while (0)
#define LT(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "%s:%d cublas %d\n", __FILE__, __LINE__, (int)s); exit(1); } } while (0)

struct Ctx {
  cublasLtHandle_t lt;
  cudaStream_t stream;
  cudaEvent_t e0, e1;
  void* ws;
  size_t ws_size = 256u << 20;
};

struct Problem {
  cublasLtMatmulDesc_t op;
  cublasLtMatrixLayout_t la, lb, lc;
  void *a, *b, *c;
  float alpha = 1.f, beta = 0.f;
};

static cublasStatus_t call(Ctx& x, Problem& p, const cublasLtMatmulAlgo_t* algo) {
  // cuBLASLt's A is our b, its B our a (see header).
  return cublasLtMatmul(x.lt, p.op, &p.alpha, p.b, p.la, p.a, p.lb, &p.beta, p.c, p.lc, p.c, p.lc, algo, x.ws,
                        x.ws_size, x.stream);
}

static float timed_once(Ctx& x, Problem& p, const cublasLtMatmulAlgo_t* algo) {
  CK(cudaEventRecord(x.e0, x.stream));
  LT(call(x, p, algo));
  CK(cudaEventRecord(x.e1, x.stream));
  CK(cudaEventSynchronize(x.e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, x.e0, x.e1));
  return ms;
}

// Warm up until the best of a block of 10 runs stops improving by >1%, then min of 30 single runs.
static double min_ns(Ctx& x, Problem& p, const cublasLtMatmulAlgo_t* algo) {
  float prev = 1e30f;
  for (int block = 0; block < 50; ++block) {
    float best = 1e30f;
    for (int i = 0; i < 10; ++i) best = std::min(best, timed_once(x, p, algo));
    if (best > prev * 0.99f && block >= 2) break;
    prev = std::min(prev, best);
  }
  float best = 1e30f;
  for (int i = 0; i < 30; ++i) best = std::min(best, timed_once(x, p, algo));
  return best * 1e6;
}

static double sustained_ns(Ctx& x, Problem& p, const cublasLtMatmulAlgo_t* algo, int reps = 200) {
  for (int i = 0; i < 20; ++i) LT(call(x, p, algo));
  CK(cudaEventRecord(x.e0, x.stream));
  for (int i = 0; i < reps; ++i) LT(call(x, p, algo));
  CK(cudaEventRecord(x.e1, x.stream));
  CK(cudaEventSynchronize(x.e1));
  float ms;
  CK(cudaEventElapsedTime(&ms, x.e0, x.e1));
  return ms * 1e6 / reps;
}

__global__ void fill(unsigned short* p, size_t n, unsigned seed, bool bf16) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned h = (unsigned)i * 2654435761u ^ seed;
    h ^= h >> 15; h *= 2246822519u; h ^= h >> 13;
    float v = ((h & 0xffff) / 65535.f) * 2.f - 1.f;  // uniform [-1, 1]
    unsigned bits;
    memcpy(&bits, &v, 4);
    // bf16: truncate f32; f16: convert.
    p[i] = bf16 ? (unsigned short)(bits >> 16) : __half_as_ushort(__float2half(v));
  }
}

int main(int argc, char** argv) {
  std::string dt = argc > 1 ? argv[1] : "bf16";
  bool bf16 = dt == "bf16";
  if (!bf16 && dt != "f16") { fprintf(stderr, "dtype: bf16|f16\n"); return 1; }
  cudaDataType_t ty = bf16 ? CUDA_R_16BF : CUDA_R_16F;
  std::vector<std::string> shapes;
  for (int i = 2; i < argc; ++i) shapes.push_back(argv[i]);
  if (shapes.empty()) shapes = {"4096x4096x4096"};

  Ctx x;
  LT(cublasLtCreate(&x.lt));
  CK(cudaStreamCreateWithFlags(&x.stream, cudaStreamNonBlocking));
  CK(cudaEventCreate(&x.e0));
  CK(cudaEventCreate(&x.e1));
  CK(cudaMalloc(&x.ws, x.ws_size));

  for (auto& s : shapes) {
    long m, n, k;
    if (sscanf(s.c_str(), "%ldx%ldx%ld", &m, &n, &k) != 3) { fprintf(stderr, "bad shape %s\n", s.c_str()); return 1; }
    Problem p;
    CK(cudaMalloc(&p.a, m * k * 2));
    CK(cudaMalloc(&p.b, n * k * 2));
    CK(cudaMalloc(&p.c, m * n * 2));
    fill<<<1024, 256, 0, x.stream>>>((unsigned short*)p.a, m * k, 1, bf16);
    fill<<<1024, 256, 0, x.stream>>>((unsigned short*)p.b, n * k, 2, bf16);
    CK(cudaStreamSynchronize(x.stream));

    LT(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof ta));
    LT(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof tb));
    LT(cublasLtMatrixLayoutCreate(&p.la, ty, k, n, k));
    LT(cublasLtMatrixLayoutCreate(&p.lb, ty, k, m, k));
    LT(cublasLtMatrixLayoutCreate(&p.lc, ty, n, m, n));

    cublasLtMatmulPreference_t pref;
    LT(cublasLtMatmulPreferenceCreate(&pref));
    LT(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &x.ws_size,
                                            sizeof x.ws_size));
    std::vector<cublasLtMatmulHeuristicResult_t> heur(64);
    int found = 0;
    LT(cublasLtMatmulAlgoGetHeuristic(x.lt, p.op, p.la, p.lb, p.lc, p.lc, pref, (int)heur.size(), heur.data(), &found));
    double flops = 2.0 * m * n * k;
    // Rounds alternate direction so no algorithm always runs right after the same neighbour or
    // only on a hotter chip; each keeps its minimum over all rounds.
    const int rounds = 3;
    std::vector<double> best_of(found, 1e30);
    for (int r = 0; r < rounds; ++r)
      for (int j = 0; j < found; ++j) {
        int i = r % 2 ? found - 1 - j : j;
        if (heur[i].state != CUBLAS_STATUS_SUCCESS || call(x, p, &heur[i].algo) != CUBLAS_STATUS_SUCCESS) continue;
        best_of[i] = std::min(best_of[i], min_ns(x, p, &heur[i].algo));
      }
    int best_i = -1;
    for (int i = 0; i < found; ++i) {
      if (best_of[i] >= 1e30) continue;
      if (best_i < 0 || best_of[i] < best_of[best_i]) best_i = i;
      if (getenv("VERBOSE")) fprintf(stderr, "  %s algo%d %.0f ns %.1f TFLOP/s\n", s.c_str(), i, best_of[i], flops / best_of[i] / 1e3);
    }
    auto report = [&](const char* tag, int i) {
      double ns = best_of[i], sus = sustained_ns(x, p, &heur[i].algo);
      printf("ref|cublaslt|%s|%ldx%ldx%ld|%s:h%d/%d|%.0f|%.1f|%.0f|%.1f\n", dt.c_str(), m, n, k, tag, i, found, ns,
             flops / ns / 1e3, sus, flops / sus / 1e3);
      fflush(stdout);
    };
    if (found > 0) report("default", 0);
    if (best_i >= 0) report("best", best_i);
    LT(cublasLtMatmulPreferenceDestroy(pref));
    cublasLtMatrixLayoutDestroy(p.la); cublasLtMatrixLayoutDestroy(p.lb); cublasLtMatrixLayoutDestroy(p.lc);
    cublasLtMatmulDescDestroy(p.op);
    CK(cudaFree(p.a)); CK(cudaFree(p.b)); CK(cudaFree(p.c));
  }
  return 0;
}
