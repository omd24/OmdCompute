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
const int   s_box_block = 32;                        // sizes the shared array, so it is fixed at compile time

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

__global__ void nlm_pad(const float* in, float* out, int S, int P, float h, float sigma_n, int W, int H)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    int v = blockIdx.y * blockDim.y + threadIdx.y;
    if (u >= W || v >= H) return;

    int R = S + P;
    int stride = W + 2 * R;                          // padded row length
    const float* p = in + (v + R) * stride + (u + R);

    float bias = 2.0f * sigma_n * sigma_n;
    float inv_h2 = 1.0f / (h * h);
    float inv_T2 = 1.0f / float((2 * P + 1) * (2 * P + 1));

    float sum = 0.0f, total = 0.0f;
    for (int dj = -S; dj <= S; ++dj)
    for (int di = -S; di <= S; ++di)
    {
        const float* q = p + dj * stride + di;
        float d2 = 0.0f;
        for (int kj = -P; kj <= P; ++kj)
        {
            const float* ra = p + kj * stride;       // row pointers replace va, vb
            const float* rb = q + kj * stride;
            for (int ki = -P; ki <= P; ++ki)
            {
                float e = ra[ki] - rb[ki];
                d2 += e * e;
            }
        }
        float wq = expf(-max(d2 * inv_T2 - bias, 0.0f) * inv_h2);
        sum += wq * q[0];
        total += wq;
    }
    out[v * W + u] = sum / total;
}

__device__ float row_sum(const float* a, int d, int P)
{
    float s = 0.0f;
    for (int ki = -P; ki <= P; ++ki)
    {
        float e = a[ki] - a[ki + d];
        s += e * e;
    }
    return s;
}

__global__ void nlm_box(const float* pad, float* out, int S, int P, float h, float sigma_n, int W, int H)
{
    __shared__ float tmp[s_box_block * (s_box_block + 2 * s_P)];   // rows -P .. block + P

    int B = blockDim.x;
    int tx = threadIdx.x, ty = threadIdx.y;
    int u = blockIdx.x * B + tx;
    int v = blockIdx.y * B + ty;

    int R = S + P;
    int stride = W + 2 * R;
    const float* a = pad + (v + R) * stride + (u + R);

    float bias = 2.0f * sigma_n * sigma_n;
    float inv_h2 = 1.0f / (h * h);
    float inv_T2 = 1.0f / float((2 * P + 1) * (2 * P + 1));

    float sum = 0.0f, total = 0.0f;                  // registers, not global buffers
    for (int dj = -S; dj <= S; ++dj)
    for (int di = -S; di <= S; ++di)
    {
        int d = dj * stride + di;

        if (u < W)                                   // the top and bottom P rows have no threads of their own
        {
            if (v < H + P)            tmp[(ty + P) * B + tx]     = row_sum(a, d, P);
            if (ty < P)               tmp[ty * B + tx]           = row_sum(a - P * stride, d, P);
            if (ty >= B - P && v < H) tmp[(ty + 2 * P) * B + tx] = row_sum(a + P * stride, d, P);
        }
        __syncthreads();                             // readers see every row of this offset

        if (u < W && v < H)
        {
            const float* t = tmp + (ty + P) * B + tx;
            float d2 = 0.0f;
            for (int kj = -P; kj <= P; ++kj) d2 += t[kj * B];
            float wq = expf(-max(d2 * inv_T2 - bias, 0.0f) * inv_h2);
            sum += wq * a[d];
            total += wq;
        }
        __syncthreads();                             // next offset must not overwrite rows still in use
    }
    if (u < W && v < H) out[v * W + u] = sum / total;
}

std::vector<float> pad_clamp(const std::vector<float>& in, int W, int H, int R)
{
    int sw = W + 2 * R, sh = H + 2 * R;
    std::vector<float> out(size_t(sw) * sh);
    for (int y = 0; y < sh; ++y)
    for (int x = 0; x < sw; ++x)
    {
        int yc = std::min(std::max(y - R, 0), H - 1);
        int xc = std::min(std::max(x - R, 0), W - 1);
        out[size_t(y) * sw + x] = in[size_t(yc) * W + xc];
    }
    return out;
}

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t0)
{
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

void compare_pad(int W, int H)
{

    const int warmup = 3, runs = 11;
    const dim3 block(32, 32);

    std::vector<float> h(size_t(W) * H);
    std::mt19937 rng(1234);
    std::normal_distribution<float> noise(0.5f, s_sigma_n);
    for (float& x : h) x = noise(rng);
    std::vector<float> hp = pad_clamp(h, W, H, s_S + s_P);

    size_t bytes = h.size() * sizeof(float), bytes_p = hp.size() * sizeof(float);
    float* d_in = nullptr; float* d_pad = nullptr; float* d_a = nullptr; float* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_pad, bytes_p));
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMemcpy(d_in, h.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pad, hp.data(), bytes_p, cudaMemcpyHostToDevice));

    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    auto clamped = [&]() {
        nlm<<<grid, block>>>(d_in, d_a, s_S, s_P, s_h, s_sigma_n, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto padded = [&]() {
        nlm_pad<<<grid, block>>>(d_pad, d_b, s_S, s_P, s_h, s_sigma_n, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    for (int r = 0; r < warmup; ++r) { clamped(); padded(); }
    std::vector<double> ratio(runs);
    for (int r = 0; r < runs; ++r)
    {
        auto t0 = Clock::now(); clamped(); double ta = ms_since(t0);
        t0 = Clock::now(); padded(); ratio[r] = ms_since(t0) / ta;
    }

    std::vector<float> a(h.size()), b(h.size());
    CUDA_CHECK(cudaMemcpy(a.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_b, bytes, cudaMemcpyDeviceToHost));
    double diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) diff = std::fmax(diff, std::fabs(double(a[i]) - b[i]));

    std::sort(ratio.begin(), ratio.end());
    printf("%4dx%-4d padded/clamped median %.3f  min %.3f  max %.3f  max diff %.2e\n",
        W, H, ratio[runs / 2], ratio[0], ratio[runs - 1], diff);

    CUDA_CHECK(cudaFree(d_in)); CUDA_CHECK(cudaFree(d_pad));
    CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
}

void compare_box(int W, int H)
{
    const int warmup = 3, runs = 11;
    const dim3 block(s_box_block, s_box_block);

    std::vector<float> h(size_t(W) * H);
    std::mt19937 rng(1234);
    std::normal_distribution<float> noise(0.5f, s_sigma_n);
    for (float& x : h) x = noise(rng);
    std::vector<float> hp = pad_clamp(h, W, H, s_S + s_P);

    size_t bytes = h.size() * sizeof(float), bytes_p = hp.size() * sizeof(float);
    float* d_pad = nullptr; float* d_a = nullptr; float* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pad, bytes_p));
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMemcpy(d_pad, hp.data(), bytes_p, cudaMemcpyHostToDevice));

    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    auto padded = [&]() {
        nlm_pad<<<grid, block>>>(d_pad, d_a, s_S, s_P, s_h, s_sigma_n, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    auto box = [&]() {
        nlm_box<<<grid, block>>>(d_pad, d_b, s_S, s_P, s_h, s_sigma_n, W, H);
        CUDA_CHECK(cudaDeviceSynchronize());
    };

    for (int r = 0; r < warmup; ++r) { padded(); box(); }
    std::vector<double> ratio(runs);
    for (int r = 0; r < runs; ++r)
    {
        auto t0 = Clock::now(); padded(); double ta = ms_since(t0);
        t0 = Clock::now(); box(); ratio[r] = ms_since(t0) / ta;
    }

    std::vector<float> a(h.size()), b(h.size());
    CUDA_CHECK(cudaMemcpy(a.data(), d_a, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), d_b, bytes, cudaMemcpyDeviceToHost));
    double diff = 0.0;
    for (size_t i = 0; i < a.size(); ++i) diff = std::fmax(diff, std::fabs(double(a[i]) - b[i]));

    std::sort(ratio.begin(), ratio.end());
    printf("%4dx%-4d box/pad median %.3f  min %.3f  max %.3f  max diff %.2e\n",
        W, H, ratio[runs / 2], ratio[0], ratio[runs - 1], diff);

    CUDA_CHECK(cudaFree(d_pad)); CUDA_CHECK(cudaFree(d_a)); CUDA_CHECK(cudaFree(d_b));
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
        auto t0 = std::chrono::steady_clock::now();
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

    compare_pad(2048, 2048);
    compare_box(2048, 2048);
}
