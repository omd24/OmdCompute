#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cassert>

/*
  NOTE:
  To use compute-sanitizer, the "Command" in Project's Properties' Debugging is set to 
  compute-sanitizer.exe at the cuda installation (at program files usually).
  Either update the Debugging's Command or just set the Command back to $(TargetPath)
  and remove the cmdarg
*/

// every CUDA API call returns an error code; this macro stops with file/line on failure
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {          \
    printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while (0)

// gpu (device) side
__global__ void add_one(float* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index = SV_DispatchThreadID.x
    if (i < n) data[i] += 1.0f;                      // the last block can extend past n
}

// cpu (host) side
int main()
{
    const int n = 1024;
    std::vector<float> h(n, 41.0f);                  // host (cpu) array

    float* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, 1024 * sizeof(float)));   // device (gpu) allocation
    CUDA_CHECK(cudaMemcpy(d, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    int block = 256;
    int grid = (n + block - 1) / block;              // ceil(n / block) = 4 blocks
    add_one<<<grid, block>>>(d, n);
    CUDA_CHECK(cudaGetLastError());                  // catches launch errors (bad config etc.)

    CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(float), cudaMemcpyDeviceToHost));  // also waits for the kernel
    CUDA_CHECK(cudaFree(d));

    printf("h[0] = %.1f, h[%d] = %.1f\n", h[0], n - 1, h[n - 1]);   // expect 42.0
    assert(h[0] == 42.0 && h[n - 1] == 42.0);
}
