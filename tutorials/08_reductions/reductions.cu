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

// First improvement: shared memory is much faster than global memory. Each block can accumulate
// partial results in isolated block-wide visible memory. This relieves the contention on a
// a single global variable that all threads want access to.

__global__ void reduce_atomic_shared(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Declare a shared float for each block.
    __shared__ float x;

    // Only one thread should initialize this shared value
    if (threadIdx.x == 0)
    {
        x = 0.0f;
    }

    // Before we continue, we must ensure that all threads can see this update by thread 0.
    __syncthreads();

    // Every thread in the block adds its input to the shared variable of the block.
    if (id < N)
    {
        atomicAdd(&x, input[id]);
    }

    // Wait until all threads have done their part
    __syncthreads();

    // Once they are all done, only one thread must add the block's partial result to the global
    // variables.
    if (threadIdx.x == 0)
    {
        atomicAdd(&d_result, x);
    }
}

// Second improvement: we can exploit the fact that the GPU is massively parallel and come up with
// a fitting procedure that uses multiple iterations. In each iteration, threads accumulate partial
// results from the previous iteration. Before, the contented access to one location forced the GPU
// to perform updates sequentially. Now, each thread can access its own, exclusive shared variable
// in each iteration in paralle.

template <unsigned int BLOCK_SIZE>
__global__ void reduce_shared(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    // Use a larger shared memory region so that each thread can store its own partial results.
    __shared__ float data[BLOCK_SIZE];

    // Use a new strategy to handle superfluous threads.
    // To make sure they stay alive and can help with the reduction, threads without an input
    // simply produce a 0, which has no effect on the result.
    data[threadIdx.x] = (id < N ? input[id] : 0);

    // In each step, a thread accumulates two partial values to form the input for the next
    // iteration. The sum of all partial results eventually yields the full result of the
    // reduction.
    for (int s = blockDim.x / 2; s > 0; s /= 2)
    {
        // In each iteration, we must make sure that all threads are done writing the updates of
        // the previous iteration / the initialization.
        __syncthreads();
        if (threadIdx.x < s)
        {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&d_result, data[0]);
    }
}

// Warp-level improvement: using warp-level primititives to accelerate the final steps of the
// reduction.
// Warps have a fast lane for communication. They are free to exchange values in registers
// when they are being scheduled for execution. 
// Warps will be formed from consecutive threads in groups of 32.
template <unsigned int BLOCK_SIZE>
__global__ void reduce_shuffle(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[BLOCK_SIZE];
    data[threadIdx.x] = (id < N ? input[id] : 0);

    // Only use shared memory until last 32 values
    for (int s = blockDim.x / 2; s > 16; s /= 2)
    {
        __syncthreads();
        if (threadIdx.x < s)
        {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
    }

    // The last 32 values can be handled with warp-level primitives
    float x = data[threadIdx.x];
    if (threadIdx.x < 32)
    {
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&d_result, x);
    }
}

// Final improvement: half of our threads actually idle after they have loaded data from global
// memory to shared! Better to have threads fetch two values at the start and then let the all
// do at least some meaningful work. 
// This means that compared to all other methods, only half the number of threads must be launched
// in the grid.
template <unsigned int BLOCK_SIZE>
__global__ void reduce_final(const float* __restrict input, int N)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ float data[BLOCK_SIZE];
    data[threadIdx.x] = id < N / 2 ? input[id] : 0;
    data[threadIdx.x] += id + N / 2 < N ? input[id + N / 2] : 0;

    for (int s = blockDim.x / 2; s > 16; s /= 2)
    {
        __syncthreads();
        if (threadIdx.x < s)
        {
            data[threadIdx.x] += data[threadIdx.x + s];
        }
    }

    float x = data[threadIdx.x];
    if (threadIdx.x < 32)
    {
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 16);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 8);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 4);
        x += __shfl_sync(0xFFFFFFFF, x, threadIdx.x + 2);
        x += __shfl_sync(0xFFFFFFFF, x, 1);
    }

    if (threadIdx.x == 0)
    {
        atomicAdd(&d_result, x);
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
        {"Atomic Shared", reduce_atomic_shared, n},
        {"Reduce Shared", reduce_shared<block_size>, n},
        {"Reduce Shuffle", reduce_shuffle<block_size>, n},
        {"Reduce Final", reduce_final<block_size>, n},
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