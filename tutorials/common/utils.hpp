#include <cuda_runtime_api.h>

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

} // namespace common