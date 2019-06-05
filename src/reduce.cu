// Includes, system
// #include <stdio.h>
// #include <stdlib.h>

// Includes, cuda
// #include <cuda_runtime.h>
// #include <cublas_v2.h>

// Includes, cuda helper functions
// #include <helper_cuda.h>

// For the functors
#include "detail/rnnt_helper.h"
#include "rnnt.h"

const int warp_size = 32;

template<int NT, typename T, typename Rop>
struct CTAReduce;

template<int NT, typename T, typename Rop>
struct CTAReduce {
    enum { Size = NT, Capacity = NT };
    struct Storage { T shared[Capacity]; };

    __device__ static T reduce(int tid, T x, Storage& storage, int count, Rop g) {
        T* s = storage.shared;
        s[tid] = x;
        __syncthreads();

        // Fold the data in half with each pass.
#pragma unroll
        for(int offset = NT / 2; offset >= warp_size; offset /= 2) {
            if(tid + offset < count && tid < offset) {
                // Read from the right half and store to the left half.
                x = g(x, s[offset + tid]);
                s[tid] = x;
            }
            __syncthreads();
        }

        T shuff;
        for (int offset = warp_size / 2; offset > 0; offset /= 2) {
            shuff = __shfl_down(x, offset);
            if (tid + offset < count && tid < offset)
                x = g(x, shuff);
        }
        return x;
    }
};

// TODO return 
template <typename T>
inline __device__ T logp(const T* ft, const T* gu, int col, int num_rows, int idx, int minibatch, int maxT, int maxU, bool batch_first) {
    int u, bt, t, mb, tu, fcol, gcol;
    if (batch_first) {
        u = col % maxU;
        bt = (col - u) / maxU;
        t = bt % maxT;
        mb = (bt - t) / maxT;
        fcol = mb * maxT + t;
        gcol = mb * maxU + u;
    } else {
        mb = col % minibatch;
        tu = (col - mb) / minibatch;
        u = tu % maxU;
        t = (tu - u) / maxU;
        fcol = t * minibatch + mb;
        gcol = u * minibatch + mb;
    }

    return ft[fcol * num_rows + idx] + gu[gcol * num_rows + idx];
}

template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_rows(Iop f, Rop g, const T* ft, const T* gu, T* output,
                            int num_rows, int num_cols, int minibatch, int maxT, int maxU, bool batch_first) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = threadIdx.x;
    int idx = tid;
    int col = blockIdx.x;
    T curr;
    T ret;

    // Each block works on a column
    if (idx < num_rows) {
        ret = logp(ft, gu, col, num_rows, idx, minibatch, maxT, maxU, batch_first);
        curr = f(ret);
    }
    idx += NT;

    while (idx < num_rows) {
        ret = logp(ft, gu, col, num_rows, idx, minibatch, maxT, maxU, batch_first);
        curr = g(curr, f(ret));
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    curr = R::reduce(tid, curr, storage, num_rows, g);

    // Store result in out
    if (tid == 0)
        output[col] = curr;
}

template <int NT, typename Iop, typename Rop, typename T>
__global__ void reduce_minus(Iop f, Rop g, const T* ft, const T* gu, T* output,
                            int num_rows, int num_cols, int minibatch, int maxT, int maxU, bool batch_first) {

    typedef CTAReduce<NT, T, Rop> R;
    __shared__ typename R::Storage storage;

    int tid = threadIdx.x;
    int idx = tid;
    int col = blockIdx.x;
    T curr;
    T ret;
    T max = output[col];

    // Each block works on a column
    if (idx < num_rows) {
        ret = logp(ft, gu, col, num_rows, idx, minibatch, maxT, maxU, batch_first);
        curr = f(ret - max);
    }
    idx += NT;

    while (idx < num_rows) {
        ret = logp(ft, gu, col, num_rows, idx, minibatch, maxT, maxU, batch_first);
        curr = g(curr, f(ret - max));
        idx += NT;
    }

    // Sum thread-totals over the CTA.
    curr = R::reduce(tid, curr, storage, num_rows, g);

    // Store result in out
    if (tid == 0)
        output[col] = -max - log(curr);
}

struct ReduceHelper {

    template<typename T, typename Iof, typename Rof>
    static void impl(Iof f, Rof g, const T* ft, const T* gu, T* output, int num_rows, int num_cols, int minibatch, int maxT, int maxU, bool minus, bool batch_first, cudaStream_t stream) {

        int grid_size;

        if (minus) {
            grid_size = num_cols;
            reduce_minus<128><<<grid_size, 128, 0, stream>>>
               (f, g, ft, gu, output, num_rows, num_cols, minibatch, maxT, maxU, batch_first);

        } else {
            grid_size = num_cols;
            reduce_rows<128><<<grid_size, 128, 0, stream>>>
               (f, g, ft, gu, output, num_rows, num_cols, minibatch, maxT, maxU, batch_first);
        }
    }
};


template<typename T, typename Iof, typename  Rof>
rnntStatus_t reduce(Iof f, Rof g, const T* ft, const T* gu, T* output, int rows, int cols, int minibatch, int maxT, int maxU, bool minus, bool batch_first, cudaStream_t stream) {
    ReduceHelper::impl(f, g, ft, gu, output, rows, cols, minibatch, maxT, maxU, minus, batch_first, stream);
    cudaStreamSynchronize(stream);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        return RNNT_STATUS_EXECUTION_FAILED;

    return RNNT_STATUS_SUCCESS;
}

rnntStatus_t reduce_exp(const float *ft, const float* gu, float *denom, int rows, int cols, int minibatch, int maxT, int maxU, bool minus, bool batch_first, cudaStream_t stream) {
    return reduce(rnnt_helper::exponential<float>(), rnnt_helper::add<float>(), ft, gu, denom, rows, cols, minibatch, maxT, maxU, minus, batch_first, stream);
}

rnntStatus_t reduce_max(const float *ft, const float* gu, float *denom, int rows, int cols, int minibatch, int maxT, int maxU, bool minus, bool batch_first, cudaStream_t stream) {
    return reduce(rnnt_helper::identity<float>(), rnnt_helper::maximum<float>(), ft, gu, denom, rows, cols, minibatch, maxT, maxU, minus, batch_first, stream);
}
