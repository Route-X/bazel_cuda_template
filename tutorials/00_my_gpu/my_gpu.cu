#include <cuda_runtime_api.h>
#include <iostream>

// Before you use your GPU to do work, you should know the most essential
// things about its capabilities.
int main()
{
    int num_devices;
    cudaGetDeviceCount(&num_devices);
    std::cout << "Num devices: " << num_devices << std::endl;

    if (num_devices == 0)
	{
		std::cout << "You have no CUDA devices available!" << std::endl;
		return -1;
	}

    // Get the ID of the currently selected active CUDA device
	int device;
	cudaGetDevice(&device);
    std::cout << "Device id: " << device << std::endl;
    // Device id: 0

    for(int i = 0; i < num_devices; ++i)
    {
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, i);

        // cudaDeviceProp contains a long range of indicators to check for different
        // things that your GPU may or may not support, as well as factors for
        // performance. 
        // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html
        //
        // However, the most essential property to know about is the compute capability
        // of the device.
        std::cout << "======== Model: " << props.name << " ========" << std::endl;
        std::cout << "Compute capability: " << props.major << "." << props.minor << std::endl;
        std::cout << "Memory: " << props.totalGlobalMem / float(1 << 30) << " GiB" << std::endl;
        std::cout << "Multiprocessors: " << props.multiProcessorCount << std::endl;
        std::cout << "Clock rage: " << props.clockRate / float(1'000'000) << " GHz" << std::endl;
    }

    return 0;
}
