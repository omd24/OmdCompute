#include "../Common/npy.h"

#include <algorithm>
#include <numeric>
#include <cmath>

// every CUDA API call returns an error code; this macro stops with file/line on failure
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {          \
    printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while (0)

// config table
const char* s_patterns[] = { "step_edge", "zone_plate" };
const int   s_sizes[][2] = { {256, 256}, {257, 129} };
const float s_sigmas[]   = { 0.5f, 1.0f, 2.0f, 4.0f };
const float s_truncate   = 3.0f;
const char* s_sigma_n    = "0.1";

// cpu-side: compute 1D gaussian weights (2R+1 weights)
std::vector<float> gaussian_kernel_1d(float sigma, float truncate)
{
    int R = (int)std::ceil(truncate * sigma); // blur radius
    std::vector<double> wd(2 * R + 1);
    double total = 0.0;
    for (int i = -R; i <= R; ++i)
    {
        wd[i + R] = std::exp(-double(i * i) / (2.0 * sigma * sigma));
        total += wd[i + R];
    }
    std::vector<float> w(2 * R + 1);
    for (int i = 0; i <= 2 * R; ++i) w[i] = float(wd[i] / total);   // normalise in double, then narrow
    return w;
}

// gpu-side: using the "in"  direct gaussian
__global__ void gaussian_direct(const float* in, float* out, const float* w, int R, int W, int H)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;   // pixel = SV_DispatchThreadID.xy
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;                    // last blocks can extend past the image

    float sum = 0.0f;
    for (int j = -R; j <= R; ++j)
    {
        int vj = min(max(v + j, 0), H - 1);          // clamp = CLAMP sampler
        for (int i = -R; i <= R; ++i)
        {
            int ui = min(max(u + i, 0), W - 1);
            sum += w[j + R] * w[i + R] * in[vj * W + ui];
        }
    }
    out[v * W + u] = sum;
}

// 
std::vector<float> filter_direct(const ImageD& y, float sigma, float truncate)
{
    std::vector<float> in(y.data.size());
    for (size_t i = 0; i < in.size(); ++i) in[i] = float(y.data[i]);  // float64 -> float32
    std::vector<float> w = gaussian_kernel_1d(sigma, truncate);       // calculate weights on cpu
    std::vector<float> out(in.size());
    int R = (int(w.size()) - 1) / 2;  // derive blur radius from w.size = 2 * R + 1
    size_t bytes = in.size() * sizeof(float);

    // fill gpu buffers with input image data and calculated weights
    float* d_in = nullptr; float* d_out = nullptr; float* d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));

    // dispatch gpu work
    dim3 block(32, 32);                              // [numthreads(32, 32, 1)]
    dim3 grid((y.W + block.x - 1) / block.x, (y.H + block.y - 1) / block.y);
    gaussian_direct<<<grid, block>>>(d_in, d_out, d_w, R, y.W, y.H);
    CUDA_CHECK(cudaGetLastError());

    // readback results (gpu-to-cpu) and return outputs
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_w));
    return out;
}

int main()
{

#pragma region Gaussian filter

    const std::string root = "../../../00-image-baseline/results/default/npy/";

    for (const char* p : s_patterns)
    for (const auto& s : s_sizes)
    for (float sigma : s_sigmas)
    {
        std::string base = root + p + "/" + std::to_string(s[0]) + "x" + std::to_string(s[1]) + "_n" + s_sigma_n;
        char suffix[64];
        snprintf(suffix, sizeof(suffix), "_gaussian_direct_s%.1f.npy", sigma);

        ImageD y   = load_npy(base + ".npy");            // noisy input
        ImageD ref = load_npy(base + suffix);            // expected float64 output
        std::vector<float> out = filter_direct(y, sigma, s_truncate);

        double max_err = 0.0;
        for (size_t i = 0; i < out.size(); ++i)
            max_err = std::fmax(max_err, std::fabs(double(out[i]) - ref.data[i]));
        printf("%-10s %dx%d s=%.1f  max|err| = %.3e\n", p, s[0], s[1], sigma, max_err);
    }

#pragma endregion

}
