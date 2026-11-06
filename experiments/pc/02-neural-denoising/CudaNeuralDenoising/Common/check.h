//CUDA_CHECK macro, event timer

#pragma once
#include <cstdio>
#include <cstdlib>

// every CUDA API call returns an error code; this macro stops with file/line on failure
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {          \
    printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while (0)
