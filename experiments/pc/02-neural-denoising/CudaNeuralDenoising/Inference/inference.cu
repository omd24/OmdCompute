#include "../Common/check.h"
#include "../Common/npy.h"
#include "../Common/conv3x3.cuh"

#include <algorithm>
#include <cmath>
#include <random>

const std::string s_dir = "export/";
const int s_layers = 13;
const int s_features = 64;
const int TP = 4, TK = 8;                            // pixels and output channels per thread in conv3x3_blocked

struct Layer
{
    int C = 0, K = 0;                                // input and output channels
    float* w = nullptr;                              // RSCK: w[((r * 3 + s) * C + c) * K + k]
    float* b = nullptr;                              // K
};

enum Mode { Naive, Blocked, Smem, Async };

const char* mode_to_str(int mode)
{
    switch (mode)
    {
        case Naive:   return "Naive";
        case Blocked: return "Blocked";
        case Smem:    return "Smem";
        case Async:    return "Async";
        default:      return "Unknown";
    }
}

// one thread per output value (pixel, channel); activations are NHWC: a[(v * W + u) * C + c]
// clamp padding, 3x3 conv, bias, then ReLU or the residual y - f(y)
__global__ void conv3x3(const float* in, const float* w, const float* b, float* out,
                        int W, int H, int C, int K, bool relu, const float* residual)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= W * H * K) return;
    int k = i % K;                                   // channel fastest: a warp shares one pixel
    int p = i / K;
    int u = p % W, v = p / W;

    float acc = b[k];
    for (int r = 0; r < 3; ++r)
    {
        int vy = min(max(v + r - 1, 0), H - 1);
        for (int s = 0; s < 3; ++s)
        {
            int ux = min(max(u + s - 1, 0), W - 1);
            const float* x = in + (size_t(vy) * W + ux) * C;   // same address across the warp: broadcast
            const float* wk = w + (r * 3 + s) * C * K + k;     // neighbouring k: coalesced
            for (int c = 0; c < C; ++c)
                acc += wk[c * K] * x[c];
        }
    }
    if (relu) acc = fmaxf(acc, 0.0f);
    out[i] = residual ? residual[p] - acc : acc;     // last layer: x = y - f(y), K = 1
}

float* upload(const std::vector<float>& h)
{
    float* d;
    CUDA_CHECK(cudaMalloc(&d, h.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
    return d;
}

std::vector<Layer> load_layers()
{
    std::vector<Layer> layers(s_layers);
    char name[64];
    for (int i = 0; i < s_layers; ++i)
    {
        sprintf(name, "conv%02d_w.npy", i);
        Array w = load_npy(s_dir + name);            // shape (3, 3, C, K)
        sprintf(name, "conv%02d_b.npy", i);
        Array b = load_npy(s_dir + name);
        layers[i].C = w.shape[2];
        layers[i].K = w.shape[3];
        layers[i].w = upload(w.data);
        layers[i].b = upload(b.data);
    }
    return layers;
}

// whole network; a and b are ping-pong buffers of W * H * features floats
// middle 64->64 layers use the kernel selected by mode; first and last layers stay naive
void forward(const std::vector<Layer>& layers, const float* y, float* x, float* a, float* b, int W, int H, Mode mode)
{
    const int block = 256;
    const float* in = y;
    for (int i = 0; i < s_layers; ++i)
    {
        const Layer& L = layers[i];
        bool last = i == s_layers - 1;
        float* out = last ? x : (i % 2 == 0 ? a : b);
        bool middle = L.C == 64 && L.K == 64;
        if (mode == Blocked && middle)
        {
            int n = (W + TP - 1) / TP * H * (64 / TK);
            conv3x3_blocked<64, 64, TP, TK><<<(n + block - 1) / block, block>>>(in, L.w, L.b, out, W, H);
        }
        else if (mode == Smem && middle)
        {
            const int PX = 64, CC = 16;              // pixels per block (one row segment), input channels per chunk
            dim3 grid((W + PX - 1) / PX, H);         // x: row segment, y: row v
            conv3x3_smem<64, 64, TP, TK, PX, CC><<<grid, (PX / TP) * (64 / TK)>>>(in, L.w, L.b, out, W, H);
        }
        else if (mode == Async && middle)
        {
            const int PX = 64, CC = 16;
            dim3 grid((W + PX - 1) / PX, H);
            conv3x3_async<64, 64, TP, TK, PX, CC><<<grid, (PX / TP) * (64 / TK)>>>(in, L.w, L.b, out, W, H);
        }
        else
        {
            int n = W * H * L.K;
            conv3x3<<<(n + block - 1) / block, block>>>(in, L.w, L.b, out, W, H, L.C, L.K, !last, last ? y : nullptr);
        }
        in = out;
    }
}

// GPU time of one forward pass
float time_once(const std::vector<Layer>& layers, const float* y, float* x, float* a, float* b, int W, int H, Mode mode)
{
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    forward(layers, y, x, a, b, W, H, mode);
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return ms;
}

// naive and blocked alternate, so both see the same clock; min of each and the ratio of the minima
void compare(const std::vector<Layer>& layers, const float* y, float* x, float* a, float* b, int W, int H, Mode m1, Mode m2, int pairs)
{
    time_once(layers, y, x, a, b, W, H, m1);      // warm-up
    time_once(layers, y, x, a, b, W, H, m2);
    float n = 1e30f, k = 1e30f;
    for (int p = 0; p < pairs; ++p)
    {
        n = std::min(n, time_once(layers, y, x, a, b, W, H, m1));
        k = std::min(k, time_once(layers, y, x, a, b, W, H, m2));
    }
    printf("%dx%d  %s %.3f ms  %s %.3f ms  ratio %.3f\n", W, H, mode_to_str(m1), n, mode_to_str(m2), k, k / n);
}

int main()
{
    std::vector<Layer> layers = load_layers();
    Array input = load_npy(s_dir + "input.npy");
    Array reference = load_npy(s_dir + "reference.npy");
    int H = input.shape[0], W = input.shape[1];

    // correctness: test crop against the PyTorch output
    float *y = upload(input.data), *x, *a, *b;
    CUDA_CHECK(cudaMalloc(&x, size_t(W) * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a, size_t(W) * H * s_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, size_t(W) * H * s_features * sizeof(float)));

    for (int m = 0; m < 4; ++m) 
    {
        forward(layers, y, x, a, b, W, H, Mode(m));
        CUDA_CHECK(cudaGetLastError());
        std::vector<float> out(size_t(W) * H);
        CUDA_CHECK(cudaMemcpy(out.data(), x, out.size() * sizeof(float), cudaMemcpyDeviceToHost));
        double err = 0.0;
        for (size_t i = 0; i < out.size(); ++i)
            err = std::max(err, double(std::fabs(out[i] - reference.data[i])));
        printf("%dx%d  %s  max abs(cuda - pytorch) %.2e\n", W, H, mode_to_str(m), err);
    }
    compare(layers, y, x, a, b, W, H, Mode::Smem, Mode::Async, 20);

    CUDA_CHECK(cudaFree(y)); CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));

    // timing at 2048x2048; content does not change the cost
    W = H = 2048;
    std::vector<float> noise(size_t(W) * H);
    std::mt19937 rng(0);
    std::normal_distribution<float> n01(0.0f, 1.0f);
    for (float& v : noise) v = 0.5f + 0.1f * n01(rng);
    y = upload(noise);
    CUDA_CHECK(cudaMalloc(&x, size_t(W) * H * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&a, size_t(W) * H * s_features * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&b, size_t(W) * H * s_features * sizeof(float)));
    compare(layers, y, x, a, b, W, H, Mode::Smem, Mode::Async, 4);
    CUDA_CHECK(cudaFree(y)); CUDA_CHECK(cudaFree(x)); CUDA_CHECK(cudaFree(a)); CUDA_CHECK(cudaFree(b));
    return 0;
}
