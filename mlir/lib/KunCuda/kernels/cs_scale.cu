// cs_scale.cu — cross-sectional scale kernel, pre-compiled to PTX and
// embedded into libKunCudaRuntime as a separate CUmodule.
//
// Signature matches cs_rank and the executor's external-kernel launch
// convention:
//   (i32 time_length, i32 num_stocks, in_ptr, out_ptr)
//
// For each timestep:
//   sum = Σ abs(x_i), ignoring NaNs
//   out_i = x_i / sum
// except all-zero valid rows follow the CPU ScaleStocks behavior and
// produce NaN for zero inputs.

#include <cuda_runtime.h>
#include <math_constants.h>

extern __shared__ unsigned char kun_cs_scale_smem[];

namespace {

template <typename T>
__device__ static inline T kun_nan();

template <>
__device__ inline float kun_nan<float>() { return CUDART_NAN_F; }

template <>
__device__ inline double kun_nan<double>() { return CUDART_NAN; }

template <typename T>
__device__ static inline T kun_abs(T v);

template <>
__device__ inline float kun_abs<float>(float v) { return fabsf(v); }

template <>
__device__ inline double kun_abs<double>(double v) { return fabs(v); }

template <typename T>
__device__ static inline T warp_sum(T v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

template <typename T>
__device__ static void cs_scale_body(const T* __restrict__ in,
                                     T* __restrict__ out,
                                     int time_length,
                                     int num_stocks) {
    int time_per_cta = (time_length + gridDim.x - 1) / gridDim.x;
    int t0 = blockIdx.x * time_per_cta;
    int t1 = t0 + time_per_cta;
    if (t1 > time_length) t1 = time_length;
    if (t0 >= t1) return;

    T* smem = reinterpret_cast<T*>(kun_cs_scale_smem);
    T* row_sum = smem + num_stocks;

    for (int t = t0; t < t1; ++t) {
        const T* row_in  = in  + static_cast<size_t>(t) * num_stocks;
        T*       row_out = out + static_cast<size_t>(t) * num_stocks;

        for (int i = threadIdx.x; i < num_stocks; i += blockDim.x) {
            smem[i] = row_in[i];
        }
        __syncthreads();

        if (threadIdx.x < 32) {
            int lane = threadIdx.x;
            T lane_sum = static_cast<T>(0);
            for (int i = lane; i < num_stocks; i += 32) {
                T v = smem[i];
                if (!isnan(v))
                    lane_sum += kun_abs(v);
            }
            T sum = warp_sum(lane_sum);
            if (lane == 0)
                *row_sum = sum;
        }
        __syncthreads();

        T sum = *row_sum;
        for (int i = threadIdx.x; i < num_stocks; i += blockDim.x) {
            T v = smem[i];
            row_out[i] = (v == static_cast<T>(0) && sum == static_cast<T>(0))
                             ? kun_nan<T>()
                             : v / sum;
        }
        __syncthreads();
    }
}

} // anonymous namespace

extern "C" __global__
void kun_cs_scale_f32(int time_length, int num_stocks,
                      const float* __restrict__ in,
                      float* __restrict__ out) {
    cs_scale_body<float>(in, out, time_length, num_stocks);
}

extern "C" __global__
void kun_cs_scale_f64(int time_length, int num_stocks,
                      const double* __restrict__ in,
                      double* __restrict__ out) {
    cs_scale_body<double>(in, out, time_length, num_stocks);
}
