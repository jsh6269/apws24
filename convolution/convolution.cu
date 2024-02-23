#include <cstdio>

#include "convolution.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

void naive_cpu_convolution(float *_I, float *_F, float *_O, int N, int C, int H,
                           int W, int K, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  float *I = _I, *F = _F, *O = _O;
  // Naive CPU convolution
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  for (int on = 0; on < ON; ++on) {
    for (int oc = 0; oc < OC; ++oc) {
      for (int oh = 0; oh < OH; ++oh) {
        for (int ow = 0; ow < OW; ++ow) {
          float sum = 0;
          for (int c = 0; c < C; ++c) {
            for (int r = 0; r < R; ++r) {
              for (int s = 0; s < S; ++s) {
                const int n = on;
                const int h = oh * stride_h - pad_h + r * dilation_h;
                const int w = ow * stride_w - pad_w + s * dilation_w;
                const int k = oc;
                if (h < 0 || h >= H || w < 0 || w >= W) continue;
                sum += I[((n * C + c) * H + h) * W + w] *
                       F[((k * C + c) * R + r) * S + s];
              }
            }
          }
          O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
        }
      }
    }
  }
}
static float *gpu_I, *gpu_F, *gpu_O;

__global__ void gpu_convolution(float *gpu_I, float *gpu_F, float *gpu_O, int N, int C, int H,
                           int W, int K, int R, int S, int pad_h, int pad_w,
                           int stride_h, int stride_w, int dilation_h,
                           int dilation_w) {
  float *I = gpu_I, *F = gpu_F, *O = gpu_O;
  // Naive CPU convolution
  const int ON = N;
  const int OC = K;
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  int temp;
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  if(tidx >= ON * OC * OH * OW){
    return;
  }

  int ow = tidx % OW;             temp = tidx / OW;
  int oh = temp % OH;             temp = temp / OH;
  int oc = temp % OC;             temp = temp / OC;
  int on = temp % ON;

  float sum = 0;
  for (int c = 0; c < C; ++c) {
    for (int r = 0; r < R; ++r) {
      for (int s = 0; s < S; ++s) {
        const int n = on;
        const int h = oh * stride_h - pad_h + r * dilation_h;
        const int w = ow * stride_w - pad_w + s * dilation_w;
        const int k = oc;
        if (h < 0 || h >= H || w < 0 || w >= W) continue;
        sum += I[((n * C + c) * H + h) * W + w] *
                F[((k * C + c) * R + r) * S + s];
      }
    }
  }
  O[((on * OC + oc) * OH + oh) * OW + ow] = sum;
}

int _ceil(int x, int y){
  return (x + y - 1) / y;
}

void convolution(float *_I, float *_F, float *_O, int N, int C, int H, int W,
                 int K, int R, int S, int pad_h, int pad_w, int stride_h,
                 int stride_w, int dilation_h, int dilation_w) {
  // Remove this line after you complete the convolution on GPU
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;
  dim3 gridDim(_ceil(N * K * OH * OW, 512));
  dim3 blockDim(512);

  cudaMemcpy(gpu_I, _I, sizeof(float)*N*C*H*W, cudaMemcpyHostToDevice);
  cudaMemcpy(gpu_F, _F, sizeof(float)*K*C*R*S, cudaMemcpyHostToDevice);

  gpu_convolution<<<gridDim, blockDim>>>(gpu_I, gpu_F, gpu_O, N, C, H, W, K, R, S, pad_h, pad_w, stride_h,
                        stride_w, dilation_h, dilation_w);

  cudaMemcpy(_O, gpu_O, sizeof(float)*N*K*OH*OW, cudaMemcpyDeviceToHost);

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_initialize(int N, int C, int H, int W, int K, int R, int S,
                            int pad_h, int pad_w, int stride_h, int stride_w,
                            int dilation_h, int dilation_w) {
  const int OH = 1 + (H + 2 * pad_h - (((R - 1) * dilation_h) + 1)) / stride_h;
  const int OW = 1 + (W + 2 * pad_w - (((S - 1) * dilation_w) + 1)) / stride_w;

  CHECK_CUDA(cudaMalloc(&gpu_I, sizeof(float) * N * C * H * W));
  CHECK_CUDA(cudaMalloc(&gpu_F, sizeof(float) * K * C * R * S));
  CHECK_CUDA(cudaMalloc(&gpu_O, sizeof(float) * N * K * OH * OW));

  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void convolution_cleanup(float *_I, float *_F, float *_O, int N, int C, int H,
                         int W, int K, int R, int S, int pad_h, int pad_w,
                         int stride_h, int stride_w, int dilation_h,
                         int dilation_w) {
  // DO NOT REMOVE; NEEDED FOR TIME MEASURE
  cudaFree(gpu_I);
  cudaFree(gpu_F);
  cudaFree(gpu_O);
  CHECK_CUDA(cudaDeviceSynchronize());
}