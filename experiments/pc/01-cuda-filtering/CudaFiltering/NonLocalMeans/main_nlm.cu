#include "../Common/npy.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <random>

// every CUDA API call returns an error code; this macro stops with file/line on failure
#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) {          \
    printf("%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } \
} while (0)

// config table
const char* s_patterns[] = { "step_edge", "zone_plate" };
const int   s_sizes[][2] = { {256, 256}, {257, 129} };
const float s_sigma_n    = 0.1f;
const int   s_S = 10, s_P = 3;                       // search and patch radius
const float s_h = 1.4f * s_sigma_n;

__global__ void nlm(const float* in, float* out, int S, int P, float h, float sigma_n, int W, int H)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    float bias = 2.0f * sigma_n * sigma_n;           // difference expected from noise alone
    float inv_h2 = 1.0f / (h * h);
    float inv_T2 = 1.0f / float((2 * P + 1) * (2 * P + 1));

    float sum = 0.0f, total = 0.0f;
    for (int dj = -S; dj <= S; ++dj)                 // candidate q = p + (di, dj)
    for (int di = -S; di <= S; ++di)
    {
        float d2 = 0.0f;
        for (int kj = -P; kj <= P; ++kj)             // compare patch(p) with patch(q)
        {
            int va = min(max(v + kj, 0), H - 1);
            int vb = min(max(v + dj + kj, 0), H - 1);
            for (int ki = -P; ki <= P; ++ki)
            {
                int ua = min(max(u + ki, 0), W - 1);
                int ub = min(max(u + di + ki, 0), W - 1);
                float e = in[va * W + ua] - in[vb * W + ub];
                d2 += e * e;
            }
        }
        float wq = expf(-max(d2 * inv_T2 - bias, 0.0f) * inv_h2);

        int vq = min(max(v + dj, 0), H - 1);
        int uq = min(max(u + di, 0), W - 1);
        sum += wq * in[vq * W + uq];                 // the patch sets the weight, q sets the value
        total += wq;
    }
    out[v * W + u] = sum / total;                    // weights are not normalised in advance
}

std::vector<float> filter_nlm(const ImageD& y, dim3 block)
{
    std::vector<float> in(y.data.size());
    for (size_t i = 0; i < in.size(); ++i) in[i] = float(y.data[i]);  // float64 -> float32
    std::vector<float> out(in.size());
    size_t bytes = in.size() * sizeof(float);

    float* d_in = nullptr; float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid((y.W + block.x - 1) / block.x, (y.H + block.y - 1) / block.y);
    nlm<<<grid, block>>>(d_in, d_out, s_S, s_P, s_h, s_sigma_n, y.W, y.H);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    return out;
}

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void time_nlm(int W, int H, dim3 block)
{
    const int warmup = 3, runs = 11;

    // noisy image: weights depend on the data, so a constant image would be a best case
    std::vector<float> h(size_t(W) * H);
    std::mt19937 rng(1234);
    std::normal_distribution<float> noise(0.5f, s_sigma_n);
    for (float& x : h) x = noise(rng);
    size_t bytes = h.size() * sizeof(float);

    float* d_in = nullptr; float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));

    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    auto launch = [&]() {
        nlm<<<grid, block>>>(d_in, d_out, s_S, s_P, s_h, s_sigma_n, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    for (int r = 0; r < warmup; ++r) launch();
    std::vector<double> t(runs);
    for (int r = 0; r < runs; ++r)
    {
        auto t0 = Clock::now();
        launch();
        t[r] = ms_since(t0);
    }
    std::sort(t.begin(), t.end());
    printf("%4dx%-4d [cpu wait] median %8.2f  min %8.2f  max %8.2f\n", W, H, t[runs / 2], t[0], t[runs - 1]);

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}

int main()
{
    const std::string root = "../../../00-image-baseline/results/default/npy/";
    const dim3 block(32, 32);

    for (const char* p : s_patterns)
    for (const auto& s : s_sizes)
    {
        std::string base = root + p + "/" + std::to_string(s[0]) + "x" + std::to_string(s[1]) + "_n0.1";
        ImageD y   = load_npy(base + ".npy");            // noisy input
        ImageD ref = load_npy(base + "_nlm.npy");        // expected float64 output
        std::vector<float> out = filter_nlm(y, block);

        double max_err = 0.0;
        for (size_t i = 0; i < out.size(); ++i)
            max_err = std::fmax(max_err, std::fabs(double(out[i]) - ref.data[i]));
        printf("%-10s %4dx%-4d  max error %.2e\n", p, s[0], s[1], max_err);
    }

    time_nlm(256, 256, block);
    time_nlm(2048, 2048, block);
}
