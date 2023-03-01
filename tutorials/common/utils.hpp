#include <cuda_runtime_api.h>
#include <random>

namespace common
{
// Define an unsigned integer variable that the GPU can work with
__device__ unsigned int step = 0;

// Increment the GPU variable N times. Whenever a thread observes
// non-consecutive numbers, it prints the latest sequence. Hence,
// every thread documents the turns that it was given by the 
// scheduler. 
static __device__ void takeNTurns(const char* who, unsigned int N)
{
    unsigned int lastTurn = 0, turn, start;
    for (int i = 0; i < N; i++)
    {
        turn = atomicInc(&step, 0xFFFFFFFFU);
        if (lastTurn != (turn-1))
            start = turn;

        if ((i == N - 1) || ((i > 0) && (start == turn)))
            printf("%s: %d--%d\n", who, start, turn);

        lastTurn = turn;
    }
}

static __global__ void testScheduling(int N)
{
    if (threadIdx.x < 2) // Branch once
        if (threadIdx.x == 0) // Branch again
            takeNTurns("Thread 1", N);
        else
            takeNTurns("Thread 2", N);
    else
        if (threadIdx.x == 2) // Branch again
            takeNTurns("Thread 3", N);
        else
            takeNTurns("Thread 4", N);
}

static void run2NestedBranchesForNSteps(int N)
{
	 testScheduling<<<1, 4>>>(N);
	 cudaDeviceSynchronize();
}


// Helper function to let threads spin
__device__ void waste_time(unsigned long long duration)
{
    const unsigned long long int start = clock64();
    while ((clock64() - start) < duration);
}

__device__ float gregory_leibniz(unsigned int iterations)
{
    float pi = 0.f, m = 1.f;
    for (int n = 0; n < iterations; n++, m *= -1.f)
    {
        pi += 4.f * (m / (2 * n + 1));
    }
    return pi;
}

static void prepare_random_numbers_cpu_gpu(unsigned int N, std::vector<float>& vals, float** dValsPtr)
{
	constexpr float target = 42.f;
	// Print expected value, because reference may be off due to floating point (im-)precision
	std::cout << "\nExpected value: " << target * N << "\n" << std::endl;

	// Generate a few random inputs to accumulate
	std::default_random_engine eng(0xcaffe);
	std::normal_distribution<float> dist(target);
	vals.resize(N);
	std::for_each(vals.begin(), vals.end(), [&dist, &eng](float& f) { f = dist(eng); });

	// Allocate some global GPU memory to write the inputs to
	cudaMalloc((void**)dValsPtr, sizeof(float) * N);
	// Expliclity copy the inputs from the CPU to the GPU
	cudaMemcpy(*dValsPtr, vals.data(), sizeof(float) * N, cudaMemcpyHostToDevice);
}
} // namespace common