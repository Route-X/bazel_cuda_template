#include <cuda_runtime_api.h>
#include <iostream>

// Define a function that will only be compiled for and called from host
__host__ void host_only()
{
    std::cout << "This function can only be called from the host!" << std::endl;
}

// Define a function that will only be compiled for and called from device
__device__ void device_only()
{
    printf("This function can only be called from the device!\n");
}

// Define a function that will be compiled for both architectures.
__host__ __device__ float square_anywhere(float x)
{
    return x * x;
}

// Define a function that will be compiled for both architectures.
__host__ __device__ void hello()
{
    # ifndef __CUDA_ARCH__
    std::cout << "Hello from CPU!" << std::endl;
    # else
    printf("Hello from GPU - CUDA ARCH: %d!\n", __CUDA_ARCH__);
    #endif
}

// Call device and portable functions from a kernel.
__global__ void run_gpu(float x)
{
    hello();
    device_only();
    printf("%f\n", square_anywhere(x));
}

// Call host and portable functions from a kernel.
// Note that, by default, if a function has no architecture specified, it is assumed to be
// __host__ by nvcc.
void run_cpu(float x)
{
    hello();
    host_only();
    std::cout << square_anywhere(x) << std::endl;
}

int main()
{
    run_cpu(42);
    run_gpu<<<1, 1>>>(42);
    cudaDeviceSynchronize();
    return 0;
}