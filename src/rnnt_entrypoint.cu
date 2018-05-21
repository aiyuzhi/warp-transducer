#include <cstddef>
#include <iostream>
#include <algorithm>

#include <rnnt.h>

#include "detail/cpu_rnnt.h"
#ifdef __CUDACC__
    #include "detail/gpu_rnnt.h"
#endif

extern "C" {

int get_warprnnt_version() {
    return 1;
}

const char* rnntGetStatusString(rnntStatus_t status) {
    switch (status) {
    case RNNT_STATUS_SUCCESS:
        return "no error";
    case RNNT_STATUS_MEMOPS_FAILED:
        return "cuda memcpy or memset failed";
    case RNNT_STATUS_INVALID_VALUE:
        return "invalid value";
    case RNNT_STATUS_EXECUTION_FAILED:
        return "execution failed";

    case RNNT_STATUS_UNKNOWN_ERROR:
    default:
        return "unknown error";

    }

}


rnntStatus_t compute_rnnt_loss(float* const trans_acts, // BTV
                             float* const pred_acts,    // BUV
                             float* trans_grad,
                             float* pred_grad,
                             const int* const flat_labels,
                             const int* const label_lengths,
                             const int* const input_lengths,
                             int alphabet_size,
                             int minibatch,
                             float *costs,
                             void *workspace,
                             rnntOptions options) {

    if (trans_acts == nullptr ||
        pred_acts == nullptr ||
        flat_labels == nullptr ||
        label_lengths == nullptr ||
        input_lengths == nullptr ||
        costs == nullptr ||
        workspace == nullptr ||
        alphabet_size <= 0 ||
        minibatch <= 0 ||
        options.maxT <= 0 ||
        options.maxU <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    if (options.loc == RNNT_CPU) {
        CpuRNNT<float> rnnt(minibatch, options.maxT, options.maxU, alphabet_size, workspace, 
                                options.blank_label, options.num_threads);

        if (trans_grad != NULL && pred_grad != NULL)
            return rnnt.cost_and_grad(trans_acts, pred_acts,
                                        trans_grad, pred_grad,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(trans_acts, pred_acts,
                                        costs, flat_labels,
                                        label_lengths, input_lengths);
    } else if (options.loc == RNNT_GPU) {
#ifdef __CUDACC__
        GpuRNNT<float> rnnt(minibatch, options.maxT, options.maxU, alphabet_size, workspace, 
                                options.blank_label, options.num_threads, options.stream);

        if (trans_grad != NULL && pred_grad != NULL)
            return rnnt.cost_and_grad(trans_acts, pred_acts,
                                        trans_grad, pred_grad,
                                        costs,
                                        flat_labels, label_lengths,
                                        input_lengths);
        else
            return rnnt.score_forward(trans_acts, pred_acts,
                                        costs, flat_labels,
                                        label_lengths, input_lengths);
#else
        std::cerr << "GPU execution requested, but not compiled with GPU support" << std::endl;
        return RNNT_STATUS_EXECUTION_FAILED;
#endif
    } else {
        return RNNT_STATUS_INVALID_VALUE;
    }
}


rnntStatus_t get_workspace_size(int maxT, int maxU,
                               int minibatch,
                               int alphabet_size,
                               bool gpu,
                               size_t* size_bytes)
{
    if (minibatch <= 0 ||
        maxT <= 0 ||
        maxU <= 0 ||
        alphabet_size <= 0)
        return RNNT_STATUS_INVALID_VALUE;

    *size_bytes = 0;

    // per minibatch memory
    size_t per_minibatch_bytes = 0;

    // alphas & betas
    per_minibatch_bytes += sizeof(float) * maxT * maxU * 2;

    // log_p, NOTE here just store the (denominator + maximum)
    // but for softmax, we need both maximum and denominator
    per_minibatch_bytes += sizeof(float) * maxT * maxU;

    if (gpu) {
        // forward-backward loglikelihood
        per_minibatch_bytes += sizeof(float) * 2;
        // labels
        per_minibatch_bytes += sizeof(int) * (maxU - 1);
        // length
        per_minibatch_bytes += sizeof(int) * 2;
    }

    *size_bytes = per_minibatch_bytes * minibatch;

    return RNNT_STATUS_SUCCESS;
}

}