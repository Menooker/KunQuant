// cs_rank.cu — cross-sectional rank kernel, pre-compiled to PTX and
// embedded into libKunCudaRuntime as a separate CUmodule.
//
// Signature matches the project's launch convention so the executor can
// pass the same `(i32 time_length, i32 num_stocks, in_ptr, out_ptr)`
// arg tuple it uses for JIT'd kernels.  Grid/block/smem are chosen by
// the executor at launch (time-major grid, one CTA per timestep,
// dynamic smem = num_stocks * sizeof(T)).
//
// Algorithm — pairwise O(N^2):
//   For each stock i with non-NaN value v,
//     less  = #{ j : !isnan(u[j]) && u[j]  < v }
//     equal = #{ j : !isnan(u[j]) && u[j] == v }    (includes i itself)
//     valid = #{ j : !isnan(u[j]) }
//   Output is the average-rank normalised to (0, 1]:
//     out = (2*less + equal + 1) / (2 * valid)
//   This matches cpp/Kun/Rank.hpp's equal_range formula exactly:
//     sum = (start + end + 1) * (end - start) / 2
//     out = sum / (end - start) / valid
//   with start = less, end = less + equal.
//
// NaN policy: NaN inputs produce NaN outputs and don't contribute to
// any count.

#include <cuda_runtime.h>
#include <math_constants.h>

// Dynamic shared memory base.  Declared at file scope (no anonymous
// namespace) so it gets a stable, internal symbol rather than nvcc's
// "extern .shared" with mangled-namespace linkage — the latter survives
// to the PTX as an unresolved extern, which the driver JIT cannot link
// when this PTX is loaded standalone via cuModuleLoadData.  Both
// kun_cs_rank_f32 and kun_cs_rank_f64 reinterpret_cast<T*>(raw_smem)
// from a single base — fine, since each kernel launch supplies its own
// physical smem allocation.
extern __shared__ unsigned char kun_cs_rank_smem[];

namespace {

template <typename T>
__device__ static inline bool kun_isnan(T x);

template <>
__device__ inline bool kun_isnan<float>(float x) { return isnan(x); }

template <>
__device__ inline bool kun_isnan<double>(double x) { return isnan(x); }

template <typename T>
__device__ static inline T kun_nan();

template <>
__device__ inline float kun_nan<float>() { return CUDART_NAN_F; }

template <>
__device__ inline double kun_nan<double>() { return CUDART_NAN; }

// Templated body — one CTA per timestep, threads cooperate across the
// cross-section.
template <typename T>
__device__ static void cs_rank_body(const T* __restrict__ in,
                                    T* __restrict__ out,
                                    int time_length,
                                    int num_stocks) {
    int t = blockIdx.x;
    if (t >= time_length) return;

    T* smem = reinterpret_cast<T*>(kun_cs_rank_smem);

    const T* row_in  = in  + static_cast<size_t>(t) * num_stocks;
    T*       row_out = out + static_cast<size_t>(t) * num_stocks;

    // 1) Cooperative load of the entire cross-section into smem.
    for (int i = threadIdx.x; i < num_stocks; i += blockDim.x) {
        smem[i] = row_in[i];
    }
    __syncthreads();

    // 2) Per-stock pairwise count.  Each thread owns a stride of stocks.
    for (int i = threadIdx.x; i < num_stocks; i += blockDim.x) {
        T v = smem[i];
        if (kun_isnan<T>(v)) {
            row_out[i] = kun_nan<T>();
            continue;
        }

        int less  = 0;
        int equal = 0;
        int valid = 0;
        for (int j = 0; j < num_stocks; ++j) {
            T u = smem[j];
            int is_valid = !kun_isnan<T>(u);
            valid += is_valid;
            less  += (is_valid & (u <  v));
            equal += (is_valid & (u == v));
        }

        if (valid == 0) {
            row_out[i] = kun_nan<T>();
            continue;
        }
        // Average-rank percentile, matching the CPU reference.
        T num = static_cast<T>(2 * less + equal + 1);
        T den = static_cast<T>(2 * valid);
        row_out[i] = num / den;
    }
}

} // anonymous namespace

extern "C" __global__
void kun_cs_rank_f32(int time_length, int num_stocks,
                     const float* __restrict__ in,
                     float* __restrict__ out) {
    cs_rank_body<float>(in, out, time_length, num_stocks);
}

extern "C" __global__
void kun_cs_rank_f64(int time_length, int num_stocks,
                     const double* __restrict__ in,
                     double* __restrict__ out) {
    cs_rank_body<double>(in, out, time_length, num_stocks);
}
