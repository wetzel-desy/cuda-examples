// This example is taken from https://devblogs.nvidia.com/even-easier-introduction-cuda/
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <sys/time.h>

// get_walltime function for time measurement
double get_walltime_(double* wcTime) {
  struct timeval tp;
  gettimeofday(&tp, NULL);
  *wcTime = (double)(tp.tv_sec + tp.tv_usec/1000000.0);
  return 0.0;
}

void get_walltime(double* wcTime) {
  get_walltime_(wcTime);
}

// CUDA Kernel function to add the elements of two arrays
__global__
void add(int n, float *x, float *y)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    //printf("Thread ID: %d \t Block ID: %d\n", threadIdx.x, blockIdx.x);
      for (int i = index; i < n; i+=stride)
              y[i] = x[i] + y[i];
}

int main(void)
{
    int N = 1<<20;
    float *x, *y;

    double delta, finish, start;
    double flops, nd;

    // Allocate Unified Memory – accessible from CPU or GPU
    cudaMallocManaged(&x, N*sizeof(float));
    cudaMallocManaged(&y, N*sizeof(float));

    // initialize x and y arrays on the host
    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Run kernel on 1M elements on the GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // start time measurement
    get_walltime(&start);

    // execute the CUDA kernel function
    add<<<numBlocks, blockSize>>>(N, x, y);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // stop time measurement, why here and not directly after the kernel call?
    get_walltime(&finish);

    // Check for errors (all values should be 3.0f)
    float maxError = 0.0f;
    for (int i = 0; i < N; i++)
        maxError = fmax(maxError, fabs(y[i]-3.0f));
    std::cout << "Max error: " << maxError << std::endl;

    // calculating time delta and Mflops
    delta = (finish - start);
    nd = (double) N;
    flops = nd/(delta * 1000000.);

    std::cout << ">>>>> finish: " << finish << std::endl;
    std::cout << ">>>>> delta: " << delta << std::endl;
    std::cout << ">>>>> Mflops: " << flops << std::endl;

    // Free memory
    cudaFree(x);
    cudaFree(y);

    return 0;
}
