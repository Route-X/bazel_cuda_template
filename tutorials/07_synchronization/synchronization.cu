#include <cuda_runtime_api.h>
#include <iostream>
#include "tutorials/common/utils.hpp"


__global__ void write_slow(int* out, int val)
{
    common::waste_time(1'000'000'000ULL);
    *out = val;    
}

__global__ void square(int* out)
{
    *out = *out * *out;
}

__global__ void approximage_pi(bool synchronized)
{
    // Create block-shared variable for approximaged pi.
    __shared__  float s_pi;
    if (threadIdx.x == 0)
    {
        s_pi = common::gregory_leibniz(100'000);
    }

    if (synchronized)
    {
        __syncthreads();
    }

    if (threadIdx.x % 32 == 0)
    {
        printf("Thread %d thinks Pi = %f\n", threadIdx.x, s_pi);
    }
}

int main()
{
    // Implicit synchronization between kernels and cudaMemcpy.
    int* d_foo_ptr;
    cudaMalloc(&d_foo_ptr, sizeof(int));

    write_slow<<<1, 1>>>(d_foo_ptr, 42);
    square<<<1, 1>>>(d_foo_ptr);

    int foo;
    cudaMemcpy(&foo, d_foo_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "42^2 = " << foo << std::endl;

    // Block-wide synchronization with syncthreads.
    approximage_pi<<<1, 128>>>(false);
    cudaDeviceSynchronize();

    approximage_pi<<<1, 128>>>(true);
    cudaDeviceSynchronize();

    return 0;
}