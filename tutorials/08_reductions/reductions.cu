#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <chrono>
#include <tuple>
#include <iomanip>
#include <utility>
#include <numeric>
#include "tutorials/common/utils.hpp"

// Declare a GPU-visible floating point variable in global memory.
__device__ float d_result;

// The most basic reduction kernel uses atomic operations to accumulate
// the individual inputs in a single, device-wide visible variable.
// NOTE: the basic atomicXXX instructions of CUDA have RELAXED semantics.
__global__ void reduce_atomic_global(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    // Since all blocks must have the same number of threads, we may have
    // to launch more threads than there are inputs.
    if (id < N)
    {
        atomicAdd(&d_result, input[id]);
    }
}

int main()
{
    constexpr unsigned int block_size = 256;
    constexpr unsigned int warmup_iterations = 10;
    constexpr unsigned int timing_iterations = 20;
    constexpr unsigned int n = 100'000'000;

    std::vector<float> vals;
    float* d_vals_ptr;
    common::prepare_random_numbers_cpu_gpu(n, vals, &d_vals_ptr);

    std::cout << "Computed CPU value: " << std::accumulate(vals.cbegin(), vals.cend(), 0.f) << std::endl;

    const std::tuple<const char*, void(*)(const float*, int), unsigned int> reduction_techniques[] 
    {
        {"Atomic Global", reduce_atomic_global, n},
        // {"Atomic Shared", reduce_atomic_shared, n},
    };
    for (const auto& [name, func, num_threads] : reduction_techniques)
    {
        // Compute the smallest grid to start required threads wtih a given block size
        const dim3 block_dim = {block_size, 1, 1};
        const dim3 grid_dim = {(num_threads + block_size - 1) / block_size, 1, 1};

        for (int i = 0; i < warmup_iterations; ++i)
        {
            func<<<grid_dim, block_dim>>>(d_vals_ptr, n);
        }

        cudaDeviceSynchronize();
        const auto before = std::chrono::system_clock::now();

        float result = 0.0f;
        for (int i = 0; i < timing_iterations; ++i)
        {
            cudaMemcpyToSymbol(d_result, &result, sizeof(float));
            func<<<grid_dim, block_dim>>>(d_vals_ptr, n);
        }

        cudaMemcpyFromSymbol(&result, d_result, sizeof(float));

        const auto after = std::chrono::system_clock::now();
        const auto elapsed = 1000.f * std::chrono::duration_cast<std::chrono::duration<float>>(after - before).count();
        std::cout << std::setw(20) << name << "\t" << elapsed / timing_iterations << "ms \t" << result << std::endl;
    }

    cudaFree(d_vals_ptr);

    return 0;
}