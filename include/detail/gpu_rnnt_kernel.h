#pragma once

#include "rnnt_helper.h"

template<typename T>
inline __device__ T logp(const T* const denom, const T* const acts, const int minibatch, const int maxT, const int maxU, const int alphabet_size, 
						int mb, int t, int u, int v, bool batch_first) {
    const int col = batch_first ? ((mb * maxT + t) * maxU + u) : ((t * maxU + u) * minibatch + mb);
    return denom[col] + acts[col * alphabet_size + v];
}

template<typename Tp>
__global__ void compute_alphas_kernel(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank, const bool batch_first) {
    // launch B blocks, each block has U threads
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    //int tid = threadIdx.x; // mb
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1); // mb label start point
    const int offset = b * maxT * maxU;
    alphas += offset;
    if (u == 0) alphas[0] = 0;

    __syncthreads();

	int n, t;
	for (n = 1; n < T+U-1; n++) {
		if (u >= U) {
        	__syncthreads();
			continue;
		}
		t = n - u;
		if (u == 0 && t < T) {
			alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, t-1, 0, blank, batch_first);
		} else if (t > 0 && t < T) {
            Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, t-1, u, blank, batch_first);
            Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, t, u-1, labels[u-1], batch_first);
            alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
		} else if (t == 0) {
            alphas[u] = alphas[u-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, 0, u-1, labels[u-1], batch_first);
		}
        __syncthreads();
	}

    __syncthreads();

    if (u == 0) {
        Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, T-1, U-1, blank, batch_first);
        llForward[b] = loglike;
    }
}

template<typename Tp>
__global__ void compute_alphas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* alphas, Tp* llForward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank, const bool batch_first) {
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
                alphas[t * maxU + u] = alphas[(t-1) * maxU + u] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, t-1, 0, blank, batch_first);
            if (t == 0 && u > 0)
                alphas[u] = alphas[u-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, 0, u-1, labels[u-1], batch_first);
            if (t > 0 && u > 0) {
                Tp no_emit = alphas[(t-1) * maxU + u] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, t-1, u, blank, batch_first);
                Tp emit = alphas[t * maxU + u-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, t, u-1, labels[u-1], batch_first);
                alphas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    Tp loglike = alphas[(T-1) * maxU + U-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, T-1, U-1, blank, batch_first);
    llForward[tid] = loglike;
}


template<typename Tp>
__global__ void compute_betas_kernel(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank, const bool batch_first) {
    int b = blockIdx.x; // batch
    int u = threadIdx.x; // label id, u
    //int tid = threadIdx.x; // mb
    const int T = xlen[b];
    const int U = ylen[b] + 1;
    const int* labels = mlabels + b * (maxU - 1);
    const int offset = b * maxT * maxU;
    betas += offset;
    if (u == 0)
        betas[(T-1) * maxU + U-1] = logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, T-1, U-1, blank, batch_first);

    __syncthreads();

	int n, t;
	for (n = T+U-3; n >= 0; n--) {
		if (u >= U) {
        	__syncthreads();
			continue;
		}

		t = n - u;
		if (u == U-1 && t >= 0) {
            betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, t, U-1, blank, batch_first);
		} else if (t >= 0 && t < T-1) {
           	Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, t, u, blank, batch_first);
            Tp emit = betas[t * maxU + u+1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, t, u, labels[u], batch_first);
            betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
        } else if (t == T-1) {
			betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, b, T-1, u, labels[u], batch_first);
		}
        __syncthreads();
	}

    __syncthreads();

    if (u == 0) {
        llBackward[b] = betas[0];
    }
}

template<typename Tp>
__global__ void compute_betas_kernel_naive(const Tp* const acts, const Tp* const denom, Tp* betas, Tp* llBackward, const int* const xlen, const int* const ylen, 
    const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_, const bool batch_first) {
    int tid = threadIdx.x; // mb
    const int T = xlen[tid];
    const int U = ylen[tid] + 1;
    const int* labels = mlabels + tid * (maxU - 1);
    const int offset = tid * maxT * maxU;
    betas += offset;
    betas[(T-1) * maxU + U-1] = logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, T-1, U-1, blank_, batch_first);

    for (int t = T-1; t >=0; --t) {
        for (int u = U-1; u >= 0; --u) {
            if (u == U-1 && t < T-1)
                betas[t * maxU + U-1] = betas[(t+1) * maxU + U-1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, t, U-1, blank_, batch_first);
            if (t == T-1 && u < U-1)
                betas[(T-1) * maxU + u] = betas[(T-1) * maxU + u+1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, T-1, u, labels[u], batch_first);
            if (t < T-1 && u < U-1) {
                Tp no_emit = betas[(t+1) * maxU + u] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, t, u, blank_, batch_first);
                Tp emit = betas[t * maxU + u+1] + logp(denom, acts, minibatch, maxT, maxU, alphabet_size, tid, t, u, labels[u], batch_first);
                betas[t * maxU + u] = rnnt_helper::log_sum_exp<Tp>(emit, no_emit);
            }
        }
    }

    llBackward[tid] = betas[0];
}

template<int NT, typename Tp>
__global__ void compute_grad_kernel(Tp* grads, const Tp* const acts, const Tp* const denom, const Tp* alphas, const Tp* betas, const Tp* const logll, const int* const xlen, const int* const ylen, const int* const mlabels, const int minibatch, const int maxT, const int maxU, const int alphabet_size, const int blank_, const bool batch_first) {
    int tid = threadIdx.x; // alphabet dim
    int idx = tid;
    int col = blockIdx.x; // mb, t, u

	int u, bt, t, mb, gcol;
    u = col % maxU;
	bt = (col - u) / maxU;
	t = bt % maxT;
	mb = (bt - t) / maxT;
    gcol = batch_first ? col : ((t*maxU+u)*minibatch+mb);

    const int T = xlen[mb];
    const int U = ylen[mb] + 1;
    const int* labels = mlabels + mb * (maxU - 1);

    if (t < T && u < U) {
        while (idx < alphabet_size) {
            Tp logpk = denom[gcol] + acts[gcol * alphabet_size + idx];
            // Tp logpk = logp(denom, acts, maxT, maxU, alphabet_size, mb, t, u, idx);
            Tp grad = exp(alphas[col] + betas[col] + logpk - logll[mb]);
            // grad to last blank transition
            if (idx == blank_ && t == T-1 && u == U-1) grad -= 1;
            if (idx == blank_ && t < T-1) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col + maxU]);
            }
            if (idx == labels[u] && u < U-1) {
                grad -= exp(alphas[col] + logpk - logll[mb] + betas[col+1]);
            }
            grads[gcol * alphabet_size + idx] = grad;

            idx += NT;
        }
    }
}
