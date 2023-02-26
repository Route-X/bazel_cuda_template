#include <cuda_runtime_api.h>
#include <iostream>

__global__ void print_ids()
{
    // Use built-in variables blockIdx and threadIdx
    const auto tID = threadIdx;
    const auto bID = blockIdx;
    printf("Block ID: (%d, %d) - Thread ID: (%d, %d)\n", bID.x, bID.y, tID.x, tID.y);
}

int main()
{
    std::cout << "Small grid: " << std::endl;
    // Configure the grid and block dimensions via built-in struct dim3 (X, Y, Z)
    const dim3 grid_size_small{1, 1, 1};
    const dim3 block_size_small{4, 4, 1};

    // Launch the kernel with custom grid
    print_ids<<<grid_size_small, block_size_small>>>();

    // Need to synchronize here to have the GPU and CPU printouts in the correct order.
    cudaDeviceSynchronize();

    std::cout << std::endl << "Large grid: " << std::endl;
    const dim3 grid_size_large{2, 2, 1};
    const dim3 block_size_large{16, 16, 1};
    print_ids<<<grid_size_large, block_size_large>>>();
    cudaDeviceSynchronize();

    std::cout << std::endl << "1D grid: " << std::endl;
    const dim3 grid_size_1d{4, 1, 1};
    const dim3 block_size_1d{4, 4, 1};
    print_ids<<<grid_size_1d, block_size_1d>>>();
    cudaDeviceSynchronize();

    std::cout << std::endl << "2D grid: " << std::endl;
    const dim3 grid_size_2d{4, 4, 1};
    const dim3 block_size_2d{4, 4, 1};
    print_ids<<<grid_size_2d, block_size_2d>>>();
    cudaDeviceSynchronize();

    std::cout << std::endl << "3D grid: " << std::endl;
    const dim3 grid_size_3d{4, 4, 4};
    const dim3 block_size_3d{4, 4, 1};
    print_ids<<<grid_size_3d, block_size_3d>>>();
    cudaDeviceSynchronize();

    return 0;
}