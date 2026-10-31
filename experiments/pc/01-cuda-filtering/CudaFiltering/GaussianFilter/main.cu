#include "../Common/npy.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>
#include <chrono>

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
const int s_blocks[][2] = { {32, 32}, {32, 16}, {16, 16}, {32, 8}, {64, 8}, {128, 4}, {32, 32} };

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
std::vector<float> filter_direct(const ImageD& y, float sigma, float truncate, dim3 block)
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
    dim3 grid((y.W + block.x - 1) / block.x, (y.H + block.y - 1) / block.y);
    gaussian_direct<<<grid, block>>>(d_in, d_out, d_w, R, y.W, y.H);
    CUDA_CHECK(cudaGetLastError());

    // readback results (gpu-to-cpu)
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    
    // free resources
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_w));

    return out;
}

__global__ void blur_h(const float* in, float* out, const float* w, int R, int W, int H)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    float sum = 0.0f;
    for (int i = -R; i <= R; ++i)
    {
        int ui = min(max(u + i, 0), W - 1);          // clamp the column only
        sum += w[i + R] * in[v * W + ui];
    }
    out[v * W + u] = sum;
}

__global__ void blur_v(const float* in, float* out, const float* w, int R, int W, int H)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    float sum = 0.0f;
    for (int j = -R; j <= R; ++j)
    {
        int vj = min(max(v + j, 0), H - 1);          // clamp the row only
        sum += w[j + R] * in[vj * W + u];
    }
    out[v * W + u] = sum;
}

std::vector<float> filter_separable(const ImageD& y, float sigma, float truncate, dim3 block)
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

    // allocate memory that holds the horizontal result
    float* d_tmp = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));    

    // dispatch gpu work
    // NOTE: No barrier is needed between the passes. 
    // Kernels launched on the same stream run in order, and each one sees the previous one's writes
    dim3 grid((y.W + block.x - 1) / block.x, (y.H + block.y - 1) / block.y);

    // horizontal blur
    blur_h<<<grid, block>>>(d_in,  d_tmp, d_w, R, y.W, y.H);

    // vertical blur
    blur_v<<<grid, block>>>(d_tmp, d_out, d_w, R, y.W, y.H);

    CUDA_CHECK(cudaGetLastError());

    // readback results (gpu-to-cpu)
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // free resources
    CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_in)); 
    CUDA_CHECK(cudaFree(d_out)); 
    CUDA_CHECK(cudaFree(d_w));

    return out;
}

//Timing
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}


float median(std::vector<float> t) // by value: sorts a copy
{
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}

// first launch (reported alone), warm-up (discarded), then steady-state runs
template <typename F>
void time_launch(const char* name, int W, int H, float sigma, F launch)
{
    const int warmup = 10, runs = 100;

    auto t0 = Clock::now();
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());     // launches are async: wait, or only the launch call is timed
    double first = ms_since(t0);

    for (int r = 0; r < warmup; ++r) { launch(); CUDA_CHECK(cudaDeviceSynchronize()); }

    std::vector<double> t(runs);
    for (int r = 0; r < runs; ++r)
    {
        t0 = Clock::now();
        launch();
        CUDA_CHECK(cudaDeviceSynchronize());
        t[r] = ms_since(t0);
    }
    std::sort(t.begin(), t.end());
    printf("%4dx%-4d s=%.1f [cpu wait] %-9s first %7.3f  median %7.3f  min %7.3f\n",
        W, H, sigma, name, first, t[runs / 2], t[0]);
}

void time_filters(int W, int H, float sigma, dim3 block)
{
    std::vector<float> h(size_t(W) * H, 0.5f);       // kernels have no data-dependent branches
    std::vector<float> w = gaussian_kernel_1d(sigma, s_truncate);
    int R = (int(w.size()) - 1) / 2;
    size_t bytes = h.size() * sizeof(float);

    float* d_in = nullptr; float* d_tmp = nullptr; float* d_out = nullptr; float* d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_tmp, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMalloc(&d_w, w.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_w, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice));

    // cpu-timing
    auto t0 = Clock::now();
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    double upload = ms_since(t0);

    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);

    time_launch("direct", W, H, sigma, [&]() {
        gaussian_direct<<<grid, block>>>(d_in, d_out, d_w, R, W, H);
    });
    time_launch("separable", W, H, sigma, [&]() {
        blur_h<<<grid, block>>>(d_in, d_tmp, d_w, R, W, H);
        blur_v<<<grid, block>>>(d_tmp, d_out, d_w, R, W, H);
    });
    
#pragma region Gpu timing: events around each pass

    cudaEvent_t e0, e1, e2;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));

    const int runs = 100;
    std::vector<float> t_dir(runs), t_sep(runs), t_h(runs), t_v(runs);
    for (int r = 0; r < runs; ++r)
    {
        CUDA_CHECK(cudaEventRecord(e0));
        gaussian_direct<<<grid, block>>>(d_in, d_out, d_w, R, W, H);
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_dir[r], e0, e1));

        CUDA_CHECK(cudaEventRecord(e0));
        blur_h<<<grid, block>>>(d_in, d_tmp, d_w, R, W, H);
        CUDA_CHECK(cudaEventRecord(e1));
        blur_v<<<grid, block>>>(d_tmp, d_out, d_w, R, W, H);
        CUDA_CHECK(cudaEventRecord(e2));
        CUDA_CHECK(cudaEventSynchronize(e2));
        CUDA_CHECK(cudaEventElapsedTime(&t_h[r], e0, e1));
        CUDA_CHECK(cudaEventElapsedTime(&t_v[r], e1, e2));
        CUDA_CHECK(cudaEventElapsedTime(&t_sep[r], e0, e2));   // whole separable filter
    }
    printf("%4dx%-4d s=%.1f [gpu time] direct %7.3f  separable %7.3f  blur_h %7.3f  blur_v %7.3f\n",
      W, H, sigma, median(t_dir), median(t_sep), median(t_h), median(t_v));

    CUDA_CHECK(cudaEventDestroy(e0));
    CUDA_CHECK(cudaEventDestroy(e1));
    CUDA_CHECK(cudaEventDestroy(e2));
#pragma endregion

    CUDA_CHECK(cudaGetLastError());

    t0 = Clock::now();
    CUDA_CHECK(cudaMemcpy(h.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    double readback = ms_since(t0);
    printf("%4dx%-4d s=%.1f [transfer] upload %7.3f  readback %7.3f\n\n", W, H, sigma, upload, readback);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_tmp));
    CUDA_CHECK(cudaFree(d_out)); CUDA_CHECK(cudaFree(d_w));
}

//
int main()
{

#pragma region Gaussian filter

    const std::string root = "../../../00-image-baseline/results/default/npy/";

    // timing loop
    printf("timings in ms\n\n");

    // using constant 0.5 image. Timings cost only depends only on W, H and R
    //for (float sigma : s_sigmas) time_filters(256, 256, sigma);

    // test various block sizes at 2k
    for (const auto& b : s_blocks)
    {
        printf("block %dx%d\n", b[0], b[1]);
        for (float sigma : s_sigmas) time_filters(2048, 2048, sigma, dim3(b[0], b[1]));
    }

    // error loop
    for (const auto& b : s_blocks)
    for (const char* p : s_patterns)
    for (const auto& s : s_sizes)
    for (float sigma : s_sigmas)
    {
        std::string base = root + p + "/" + std::to_string(s[0]) + "x" + std::to_string(s[1]) + "_n" + s_sigma_n;
        char suffix[64];
        snprintf(suffix, sizeof(suffix), "_gaussian_direct_s%.1f.npy", sigma);

        ImageD y   = load_npy(base + ".npy");            // noisy input
        ImageD ref = load_npy(base + suffix);            // expected float64 output
        std::vector<float> dir_out = filter_direct(y, sigma, s_truncate, dim3(b[0], b[1]));
        std::vector<float> sep_out = filter_separable(y, sigma, s_truncate, dim3(b[0], b[1]));
        assert(dir_out.size() == sep_out.size());

        // compare and output results:
        double max_err_dir = 0.0, max_err_sep = 0.0, diff = 0.0;
        for (size_t i = 0; i < dir_out.size(); ++i)
        {
            max_err_dir = std::fmax(max_err_dir, std::fabs(double(dir_out[i]) - ref.data[i]));
            max_err_sep = std::fmax(max_err_sep, std::fabs(double(sep_out[i]) - ref.data[i]));
            diff = std::fmax(diff, std::fabs(double(dir_out[i]) - double(sep_out[i])));
        }
        printf("%-10s %4dx%-4d s=%.1f  direct %.2e  separable %.2e  diff %.2e\n",
            p, s[0], s[1], sigma, max_err_dir, max_err_sep, diff);
    }
    
#pragma endregion

}
