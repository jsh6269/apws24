#define THREADS_PER_BLOCK 1024
#define ELEMENTS_PER_BLOCK (THREADS_PER_BLOCK * 2)

__global__ void reduce_kernel(double *input, double *output, int N){
    extern __shared__ double L[];
    unsigned int tid = threadIdx.x;
    unsigned int offset = blockIdx.x * blockDim.x * 2;
    unsigned int stride = blockDim.x;

    L[tid] = 0;
    if(tid + offset < N){
        L[tid] += input[tid + offset];
    }
    if(tid + stride + offset < N){
        L[tid] += input[tid + stride + offset];
    }
    __syncthreads();

    for(stride = blockDim.x / 2; stride > 0; stride /= 2){
        if(tid < stride){
            L[tid] += L[tid + stride];
        }
        __syncthreads();
    }
    if(tid == 0){
        output[blockIdx.x] = L[0];
    }
}

double reducion_gpu(double* A, size_t num_elements){
    size_t output_elements = (num_elements + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    cudaMemcpy(input_gpu, A, num_elements * sizeof(double), cudaMemcpyHostToDevice);

    dim3 gridDim(output_elements);
    dim3 blockDim(THREADS_PER_BLOCK);
    reduce_kernel<<<gridDim, blockDim, THREADS_PER_BLOCK * sizeof(double), 0>>>(input_gpu, output_gpu, num_elements);

    double sum = 0.0;
    cudaMemcpy(output_cpu, output_gpu, output_elements * sizeof(double), cudaMemcpyDeviceToHost);
    for(size_t i=0; i<output_elements; i++){
        sum += output_cpu[i];
    }
    return sum;
}