#pragma once

rnntStatus_t reduce_exp(const float* ft, const float* gu, float* denom, int rows, int cols, int minibatch, int maxT, int maxU, bool minus, bool batch_first, cudaStream_t stream);
rnntStatus_t reduce_max(const float* ft, const float* gu, float* denom, int rows, int cols, int minibatch, int maxT, int maxU, bool minus, bool batch_first, cudaStream_t stream);
