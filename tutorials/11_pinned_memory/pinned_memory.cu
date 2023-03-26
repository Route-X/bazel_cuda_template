#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include "tutorials/common/utils.hpp"

__global__ void perform_complex_task(float input, float* __restrict result)
{
    common::waste_time(100'000'000ULL);
    *result = input * input;
}

int main()
{
    // Pinned memory becomes relevant once we start using streams and memory transfer
    // the mix. The default memcpy operation cudaMemcpy is, by default synchronous, i.e.,
    // when it is called, the CPU will stall until the memcpy has finished. However, in
    // many cases we don't want this. 
    //
    // Ideally, we would like the memory transfers to overlap with kernels that run in different
    // streams. But if we use cudaMemcpy, the kernel calls will execute sequentially, because
    // each cudaMemcpy implicitly synchronizes the default stream with the CPU, and all basic
    // streams are synchronized with the default stream. 
    // To perform asynchronous memcpy between the device and the host, CUDA must be sure that
    // the host memory is available in main memory. We can guarantee this by allocating memory
    // with cudaMallocHost. This is so-called "pinned" memory, which may never be moved or
    // swapped out. 
    // If we use pinned memory and cudaMemcpyAsync, then copies and kernels that run in different
    // streams are free to overlap.

    constexpr unsigned int TASKS = 4;

    // Allocate result values for GPU to write to
    float* d_results_ptr;
    cudaMalloc((void**)&d_results_ptr, sizeof(float) * TASKS);

    // Generate necessary streams and events
    cudaStream_t streams[TASKS];
    cudaEvent_t events[TASKS];
    for (int i = 0; i < TASKS; ++i)
    {
        cudaStreamCreate(&streams[i]);
        cudaEventCreate(&events[i]);
    }

    float results[TASKS], *results_pinned;
    cudaMallocHost((void**)&results_pinned, sizeof(float) * TASKS);

    // We run the tasks with regular/async memcpy
    enum class CPYTYPE { MEMCPY, MEMCPYASYNC };
    // We run the tasks with regular/pinned memory
    enum class MEMTYPE { REGULAR, PINNED };

    for (auto cpy : { CPYTYPE::MEMCPY, CPYTYPE::MEMCPYASYNC })
    {
        for (auto mem : { MEMTYPE::REGULAR, MEMTYPE::PINNED })
        {
            float* dst = (mem == MEMTYPE::PINNED ? results_pinned : results);

            std::cout << "Performing tasks with " << (mem == MEMTYPE::PINNED ? "pinned memory" : "regular memory");
            std::cout << " and " << (cpy == CPYTYPE::MEMCPYASYNC ? "asynchronous" : "regular") << " copy" << std::endl;

            cudaMemset(d_results_ptr, 0, sizeof(float) * TASKS);
            cudaDeviceSynchronize();
            const auto before = std::chrono::system_clock::now();

            for (int i = 0; i < TASKS; ++i)
            {
                perform_complex_task<<<1, 1, 0, streams[i]>>>(i + 1, d_results_ptr + i);
                if (cpy == CPYTYPE::MEMCPYASYNC)
                {
                    cudaMemcpyAsync(&dst[i], d_results_ptr + i, sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
                }
                else
                {
                    cudaMemcpy(&dst[i], d_results_ptr + i, sizeof(float), cudaMemcpyDeviceToHost);
                }
            }

            for (int i = 0; i < TASKS; ++i)
            {
                cudaStreamSynchronize(streams[i]);
                if (dst[i] != (i + 1) * (i + 1))
                {
                    std::cout << "Task failed or CPU received wrong value!" << std::endl;
                }
                else
                {
                    std::cout << "Finished task " << i << ", produced output: " << results[i] << std::endl;
                }
            }

            const auto after = std::chrono::system_clock::now();
            std::cout << "Time: " << std::chrono::duration_cast<std::chrono::duration<float>>(after - before).count() << "s\n\n";
        }
    }

    for (cudaStream_t& s : streams)
    {
        cudaStreamDestroy(s);
    }

    // Pinned memory should be freed with cudaFreeHost.
    cudaFreeHost(results_pinned);
}