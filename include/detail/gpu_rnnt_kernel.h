#pragma once

#include "rnnt_helper.h"

template<typename T>
inline __device__ T logp(const T* const denom, const T* const trans_acts, const T* const pred_acts, const int minibatch, const int maxT, const int maxU, 
                        const int alphabet_size, int mb, int t, int u, int v, bool batch_first) {
    const int col = batch_first ? ((mb * maxT + t) * maxU + u) : ((t * maxU + u) * minibatch + mb);
    const int tcol = batch_first ? (mb * maxT + t) : (t * minibatch + mb);
    const int pcol = batch_first ? (mb * maxU + u) : (u * minibatch + mb);
    return denom[col] + trans_acts[tcol * alphabet_size + v] + pred_acts[pcol * alphabet_size + v];
}

template<typename Tp>
__global__ void compute_alphas_kernel(const Tp* const trans_acts, const Tp* const pred_acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank, const bool batch_first) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1); // mb label start point
    const int offset = tid * maxT * maxU;
    alphas += offset;
    alphas[0] = 0;

    for (int t = 0; t < T; ++t) {
        for (int u = 0; u < U; ++u) {
            if (u == 0 && t > 0)
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, t-1, 0, blank, batch_first);
            if (t == 0 && u > 0)
                alphas[u] = alphas[u-1] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, 0, u-1, labels[u-1], batch_first);
            if (t > 0 && u > 0) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, t-1, u, blank, batch_first);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, t, u-1, labels[u-1], batch_first);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, T-1, U-1, blank, batch_first);
    llForward[tid] = loglike;
    __syncthreads();
}

template<typename Tp>
__global__ void compute_betas_kernel(const Tp* const trans_acts, const Tp* const pred_acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank, const bool batch_first) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1);
    const int offset = tid * maxT * maxU;
    betas += offset;
    betas[(T-1) * maxU + U-1] = logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, T-1, U-1, blank, batch_first);

    for (int t = T-1; t >=0; --t) {
        for (int u = U-1; u >= 0; --u) {
            if (u == U-1 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, t, U-1, blank, batch_first);
            if (t == T-1 && u < U-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, T-1, u, labels[u], batch_first);
            if (t < T-1 && u < U-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, t, u, blank, batch_first);
                Tp emit = betas[t * maxU + u+1] + logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, tid, t, u, labels[u], batch_first);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    llBackward[tid] = betas[0];
    __syncthreads();
}

template<int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* trans_grad, Tp* pred_grad, const Tp* const trans_acts, const Tp* const pred_acts, const Tp* const denom, const Tp* const alphas, const Tp* const betas, const Tp* const logll, const int* const xlen, const int* const ylen, const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank, const bool batch_first) {
    int tid = threadIdx.x; // alphabet dim
    int idx = tid;
    int col = blockIdx.x; // mb, t, u

    int u = col % maxU;
    int bt = (col - u) / maxU;
    int t = bt % maxT;
    int mb = (bt - t) / maxT;
    const int tcol = batch_first ? (mb * maxT + t) : (t * minibatch + mb);
    const int pcol = batch_first ? (mb * maxU + u) : (u * minibatch + mb);

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = logp(denom, trans_acts, pred_acts, minibatch, maxT, maxU, alphabet_size, mb, t, u, idx, batch_first);
            Tp grad = exp(alphas[col] + betas[col] + logpk - logll[mb]);
            // grad to last blank transition
            if (idx == blank && t == T-1 && u == U-1) grad -= 1;
            if (idx == blank && t < T-1) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + maxU]);
            }
            if (idx == labels[u] && u < U-1) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col+1]);
            }
            atomicAdd(&trans_grad[tcol * alphabet_size + idx], grad);
            atomicAdd(&pred_grad[pcol * alphabet_size + idx], grad);

            idx += NT;
        }
    }
    __syncthreads();
}
