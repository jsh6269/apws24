#include <cstdio>

__global__ void hello_world() {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  printf("Device(GPU) Thread %d: Hello, World!\n", tidx);
}

int main()
{
  hello_world<<<3, 5>>>();
  cudaDeviceSynchronize();
  return 0;
}
