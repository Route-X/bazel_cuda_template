#include <cuda_runtime_api.h>
#include <iostream>
#include "tutorials/common/utils.hpp"

int main()
{
    constexpr int N = 256;
    // Using a utility function for demonstration
    common::run2NestedBranchesForNSteps(N);
    return 0;
}