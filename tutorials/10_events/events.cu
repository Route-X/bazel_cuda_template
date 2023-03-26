#include <cuda_runtime_api.h>
#include <iostream>
#include <chrono>
#include <thread>
#include "tutorials/common/utils.hpp"

// A kernel that wastes some time
__global__ void slow_kernel()
{
    common::waste_time(1'000'000'000ULL);
}

__device__ int d_foo;

// A kernel that only sets d_foo
__global__ void set_foo(int foo)
{
    d_foo = foo;
}

// A kernel that prints d_foo
__global__ void print_foo()
{
    printf("foo: %d\n", d_foo);
}

int main()
{
    // Use events to measure time and communicate accross streams
    using namespace std::chrono_literals;
    using namespace std::chrono;

    // Create CUDA events
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Synchronize GPU with CPU to capture adequate time
    cudaDeviceSynchronize();
    auto before = std::chrono::system_clock::now();

    // Record start directly before first relevant GPU command
    cudaEventRecord(start);

    // Launch a light-weight GPU kernel and heavy GPU kernel
    set_foo<<<1, 1>>>(0);
    slow_kernel<<<1, 1>>>();

    // Record end directly after last relevant GPU command
    cudaEventRecord(end);
    // Also measure CPU time after laster GPU command, without synchronizing.
    auto after_no_sync = std::chrono::system_clock::now();

    // Synchronize CPU and GPU
    cudaDeviceSynchronize();
    // Measure CPU time after last GPU command, with synchronizing.
    auto after_sync = std::chrono::system_clock::now();

    // Print measured CPU time without synchronization
    float ms_cpu_no_sync = 1000.f * duration_cast<duration<float>>(after_no_sync - before).count();
    std::cout << "Measured time (chrono, no sync): " << ms_cpu_no_sync << "ms\n";

    // Print measured CPU time with synchronization
    float ms_cpu_sync = 1000.f * duration_cast<duration<float>>(after_sync - before).count();
    std::cout << "Measured time (chrono, sync): " << ms_cpu_sync << "ms\n";

    // Print measured GPU time measured with CUDA events
    float ms_gpu;
    cudaEventElapsedTime(&ms_gpu, start, end);
    std::cout << "Measured time (CUDA events): " << ms_gpu << "ms\n";

    // The difference between the two methods, CPU timing and events, is important when
    // writing more complex projects: kernels are being launched asynchronously. 
    // With CUDA events, we can insert them into streams before and after the actions
    // we want to measure.
    
    cudaEventDestroy(start);
    cudaEventDestroy(end);

    // Events may also be used to introduce dependencies across streams.
    // One stream may compute an important piece of information that another should use.
    // This dependency can be modelled by recording an event in one stream and have the
    // target stream wait on this event.
    
    cudaEvent_t foo_ready;
    cudaEventCreate(&foo_ready);

    cudaStream_t producer, consumer;
    cudaStreamCreate(&producer);
    cudaStreamCreate(&consumer);

    slow_kernel<<<1, 1, 0, producer>>>();
    set_foo<<<1, 1, 0< producer>>>(42);
    // Producer notifies consumer stream that foo is ready
    cudaEventRecord(foo_ready, producer);

    cudaStreamWaitEvent(consumer, foo_ready);
    print_foo<<<1, 1, 0, consumer>>>();

    cudaDeviceSynchronize();

    return 0;
}