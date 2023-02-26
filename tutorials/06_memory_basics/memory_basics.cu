#include <cuda_runtime_api.h>
#include <iostream>


// Declare a devcie variable in constant memory
__constant__ int c_foo;

__global__ void read_constant_memory()
{
    printf("GPU: reading constant memory --> %x\n", c_foo);
}

// Declare a device variable in global memory
__device__ const int d_foo = 42;

__global__ void read_global_memory(const int* __restrict d_bar_ptr)
{
    printf("GPU: reading global memory --> %d %x\n", d_foo, *d_bar_ptr);
}

__global__ void write_global_memory(int* __restrict d_output_ptr)
{
    *d_output_ptr = d_foo * d_foo;
}

__device__ void write_and_print_shared_memory(int* s_foo)
{
    // Write a computed result to shared memory for other threads to see
    s_foo[threadIdx.x] = 42 * (threadIdx.x + 1);
    // We make sure that no thread prints while the other still writes (parallelism!)
    __syncwarp();
    // Print own computed result and result by neighbor
    printf("ThreadID: %d, s_foo[0]: %d, s_foo[1]: %d\n", threadIdx.x, s_foo[0], s_foo[1]);
}

__global__ void write_and_print_shared_memory_fixed()
{
    // Fixed allocation of two integers in shared memory
    __shared__ int s_foo[2];
    write_and_print_shared_memory(s_foo);
}

__global__ void write_and_print_shared_memory_dynamic()
{
    // Use dynamically allocated shared memory
    extern __shared__ int s_foo[];
    write_and_print_shared_memory(s_foo);
}

int main()
{
    const int bar = 0xcaffe;
    // Uniform variables should best be placed in const GPU memory.
    // Can be updated with cudaMemcpyToSymbol.
    // This syntax is unusual, but this is how it should be.
    cudaMemcpyToSymbol(c_foo, &bar, sizeof(int));
    read_constant_memory<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Larger or read-write data is easiest provisioned by global memory.
    // Can be allocated with cudaMalloc and updated with cudaMemcpy.
    // Must be freed afterward.
    int* d_bar_ptr;
    cudaMalloc((void**)&d_bar_ptr, sizeof(int));
    cudaMemcpy(d_bar_ptr, &bar, sizeof(int), cudaMemcpyHostToDevice);
    read_global_memory<<<1, 1>>>(d_bar_ptr);
    cudaDeviceSynchronize();
    cudaFree(d_bar_ptr);

    // The CPU may also read back updates from the GPU by copying the relevant data from
    // global memory after running the kernel.
    // Notice that we do not use cudaDeviceSynchronize: cudaMemcpy will synchronize
    // with the CPU automatically.
    int out, *d_output_ptr;
    cudaMalloc((void**)&d_output_ptr, sizeof(int));
    write_global_memory<<<1, 1>>>(d_output_ptr);
    cudaMemcpy(&out, d_output_ptr, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_output_ptr);
    std::cout << "CPU: copied back from GPU --> " << out << std::endl;

    // For information that is shared only within a single threadblock, we can also use
    // shared memory, which is usually more efficient than global memory.
    // Shared memory for a block may be 
    //  - statically allocated inside the kernel,
    //  - dynamically allocated at the kernel launch.
    // In the latter case, the size of the required memory is provided as the third
    // launch parameter, and the kernel will be able to access the allocated shared
    // memory via an array with the "extern" decoration.
    std::cout << "Using static shared memory" << std::endl;
    write_and_print_shared_memory_fixed<<<1, 2>>>();
    cudaDeviceSynchronize();

    std::cout << "Using dynamic shared memory" << std::endl;
    write_and_print_shared_memory_dynamic<<<1, 2, 2 * sizeof(int)>>>();
    cudaDeviceSynchronize();

    return 0;
}
