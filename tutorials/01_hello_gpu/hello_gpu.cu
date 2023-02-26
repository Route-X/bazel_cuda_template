#include <cuda_runtime_api.h>
#include <iostream>

// Hello kernel.
__global__ void hello_gpu()
{
    printf("Hello from the GPU!\n");
}

// World kernel.
__global__ void world_gpu()
{
    printf("World from the GPU!\n");
}

int main()
{
    // Launch the kernel with 1 block that has 12 threads.
    hello_gpu<<<1, 12>>>();
    // Launch the kernel with 1 block that has 6 threads.
    world_gpu<<<1, 6>>>();
    world_gpu<<<1, 6>>>();

    // Synchronize with GPU to wait for printf to finish.
    // Results of printf are buffered and copied back to the GPU for I/O after
    // the kernel has finished.
    cudaDeviceSynchronize();

    return 0;
}