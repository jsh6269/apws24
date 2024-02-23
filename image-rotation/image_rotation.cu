#include <cstdio>

#include "image_rotation.h"

#define CHECK_CUDA(call)                                                 \
  do {                                                                   \
    cudaError_t status_ = call;                                          \
    if (status_ != cudaSuccess) {                                        \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__, \
              cudaGetErrorName(status_), cudaGetErrorString(status_));   \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)

// Device(GPU) pointers
static float *input_images_gpu, *output_images_gpu;
static int *dW, *dH, *dnum_src_images;
static float *dsin_theta, *dcos_theta;

void rotate_image_naive(float *input_images, float *output_images, int W, int H,
                        float sin_theta, float cos_theta, int num_src_images) {
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  // Rotate images
  for (int i = 0; i < num_src_images; i++) {
    for (int dest_x = 0; dest_x < W; dest_x++) {
      for (int dest_y = 0; dest_y < H; dest_y++) {
        float xOff = dest_x - x0;
        float yOff = dest_y - y0;
        int src_x = (int) (xOff * cos_theta + yOff * sin_theta + x0);
        int src_y = (int) (yOff * cos_theta - xOff * sin_theta + y0);
        if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
          output_images[i * H * W + dest_y * W + dest_x] =
              input_images[i * H * W + src_y * W + src_x];
        } else {
          output_images[i * H * W + dest_y * W + dest_x] = 0.0f;
        }
      }
    }
  }
}

__global__ void rotate_image_gpu(float *input_images, float *output_images, int* dW, int* dH,
                  float* sin_theta, float* cos_theta, int* dnum_src_images) {
  int tidx = blockIdx.x * blockDim.x + threadIdx.x;
  int tidy = blockIdx.y * blockDim.y + threadIdx.y;
  int dest_x = tidx;
  int dest_y = tidy;
  int W = *dW;
  int H = *dH;
  int num_src_images = *dnum_src_images;
  float x0 = W / 2.0f;
  float y0 = H / 2.0f;

  if(tidx >= W || tidy >= H)
    return;

  for (int i = 0; i < num_src_images; i++) {
      float xOff = dest_x - x0;
      float yOff = dest_y - y0;
      int src_x = (int) (xOff * *cos_theta + yOff * *sin_theta + x0);
      int src_y = (int) (yOff * *cos_theta - xOff * *sin_theta + y0);
      if ((src_x >= 0) && (src_x < W) && (src_y >= 0) && (src_y < H)) {
        output_images[i * H * W + dest_y * W + dest_x] =
            input_images[i * H * W + src_y * W + src_x];
      } else {
        output_images[i * H * W + dest_y * W + dest_x] = 0.0f;
      }
  }
    
}

int _ceil(int x, int y){
  return (x + y - 1) / y;
}

void rotate_image(float *input_images, float *output_images, int W, int H,
                  float sin_theta, float cos_theta, int num_src_images) {
  // Remove this line after you complete the image rotation on GPU
  rotate_image_naive(input_images, output_images, W, H, sin_theta, cos_theta,
                     num_src_images);

  // (TODO) Upload input images to GPU
  CHECK_CUDA(cudaMemcpy(dW, &W, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dH, &H, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dnum_src_images, &num_src_images, sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dcos_theta, &cos_theta, sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dsin_theta, &sin_theta, sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(input_images_gpu, input_images, sizeof(float) * (W * H), cudaMemcpyHostToDevice));

  // (TODO) Launch kernel on GPU
  dim3 gridDim(_ceil(W, 16), _ceil(H, 16));
  dim3 blockDim(16, 16);
  rotate_image_gpu<<<gridDim, blockDim>>>(input_images_gpu, output_images_gpu, dW, dH, dsin_theta, dcos_theta, dnum_src_images);

  // (TODO) Download output images from GPU
  CHECK_CUDA(cudaMemcpy(output_images, output_images_gpu, sizeof(float) * (W * H), cudaMemcpyDeviceToHost));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void rotate_image_init(int image_width, int image_height, int num_src_images) {
  // (TODO) Allocate device memory
  // static float *input_images_gpu, *output_images_gpu;
  // static int *dW, *dH, *dnum_src_images;
  // static float *dsin_theta, *dcos_theta;

  CHECK_CUDA(cudaMalloc(&input_images_gpu, sizeof(float) * image_width * image_height));
  CHECK_CUDA(cudaMalloc(&output_images_gpu, sizeof(float) * image_width * image_height));
  CHECK_CUDA(cudaMalloc(&dsin_theta, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dcos_theta, sizeof(float)));
  CHECK_CUDA(cudaMalloc(&dW, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dH, sizeof(int)));
  CHECK_CUDA(cudaMalloc(&dnum_src_images, sizeof(int)));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}

void rotate_image_cleanup() {
  // (TODO) Free device memory
  CHECK_CUDA(cudaFree(input_images_gpu));
  CHECK_CUDA(cudaFree(output_images_gpu));
  CHECK_CUDA(cudaFree(dsin_theta));
  CHECK_CUDA(cudaFree(dcos_theta));
  CHECK_CUDA(cudaFree(dW));
  CHECK_CUDA(cudaFree(dH));
  CHECK_CUDA(cudaFree(dnum_src_images));

  // DO NOT REMOVE; NEED FOR TIME MEASURE
  CHECK_CUDA(cudaDeviceSynchronize());
}
