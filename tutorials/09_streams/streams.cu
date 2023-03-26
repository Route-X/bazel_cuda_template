#include <cuda_runtime_api.h>
#include <iostream>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include "tutorials/common/utils.hpp"

__global__ void busy()
{
    common::waste_time(1'000'000'000ULL);
    printf("I'm awake!\n");
}

constexpr unsigned int KERNEL_CALLS = 2;

int main()
{
    std::cout << "Running sequential launches" << std::endl;
    for (unsigned int i = 0; i < KERNEL_CALLS; ++i)
    {
        busy<<<1, 1>>>();
    }
    // Synchronize before continuing to get clear separation in Nsight.
    cudaDeviceSynchronize();

    std::cout << "\nRunning launches in streams" << std::endl;

    // Allocate one stream for each kernel to be launched.
    cudaStream_t streams[KERNEL_CALLS];
    for (cudaStream_t& s : streams)
    {
        // Create stream and launch kernel into it
        cudaStreamCreate(&s);
        busy<<<1, 1, 0, s>>>();
    }

    // Destroy all streams. It is fine to do that immediately. 
    // Will not implicitly synchronize, but the GPU will continue running their jobs until they
    // have all been taken care of
    for (cudaStream_t& s : streams)
    {
        cudaStreamDestroy(s);
    }
    cudaDeviceSynchronize();

    // If we don't specify a stream, then the kernel is launched into the default stream.
    // Also, many operations like cudaDeviceSynchronize and cudaStreamSynchronize are submitted
    // to the default stream. Usually, only a single default stream is defined per application,
    // meaning that if you don't specify streams, you will not be able to benefit from kernels
    // running concurrently. Hence, any elaborate CUDA application should be using streams.

    // However, if the task can be cleanly separated into CPU threads, there is another option:
    // using per-thread default streams. Each thread will use its own defautl stream if we
    // pass the built-in value cudaStreamPerThread as the stream to use.
    // Kernels can then run concurrently on the GPU by creating multiple CPU threads.
    // Alternatively, you may set the compiler option "--default-stream per-thread".
    // This way, CPU threads will use separate default streams if none are specified.
    std::cout << "\nRunning threads with different default streams" << std::endl;

    // Create mutex, condition variable and kernel counter for communication
    std::mutex mutex;
    std::condition_variable cv;
    unsigned int kernels_launched = 0;

    // Allocate sufficient number of threads.
    std::thread threads[KERNEL_CALLS];

    // Create a separate thread for each kernel call.
    for (std::thread& t : threads)
    {
        t = std::thread([&mutex, &cv, &kernels_launched] {
            // Launch kernel to thread's default stream
            busy<<<1, 1, 0, cudaStreamPerThread>>>();

            std::unique_lock<std::mutex> lock(mutex);
            ++kernels_launched;
            cv.wait(lock, [&kernels_launched] { return kernels_launched == KERNEL_CALLS; });
            cv.notify_all();

            // Synchronize to wait for printf output
            cudaStreamSynchronize(cudaStreamPerThread);
        });
    }

    // Wait for all threads to finish launching their kernels in individual streams
    std::for_each(threads, threads + KERNEL_CALLS, [](std::thread& t) {t.join(); });

    // By default, custom created streams will implicitly synchronize with the default
    // stream. 
    cudaStream_t custom_regular, custom_non_blocking;
    cudaStreamCreate(&custom_regular);
    cudaStreamCreateWithFlags(&custom_non_blocking, cudaStreamNonBlocking);

    auto testAB = [](const char* kind, cudaStream_t stream) {
        std::cout << "\nLaunching A (custom) -> B (default) with " << kind 
                  << " custom stream" << std::endl;
        busy<<<1, 1, 0, stream>>>();
        busy<<<1, 1>>>();
        cudaDeviceSynchronize();
    };

    testAB("regular", custom_regular);
    testAB("non-blocking", custom_non_blocking);

    // Clean up generated streams
    cudaStreamDestroy(custom_regular);
    cudaStreamDestroy(custom_non_blocking);

    return 0;
}