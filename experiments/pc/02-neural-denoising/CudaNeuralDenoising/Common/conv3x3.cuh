#pragma once

__device__ __forceinline__ float lane(const float4& a, int i)  // i is a compile-time constant after unrolling
{
    return i == 0 ? a.x : i == 1 ? a.y : i == 2 ? a.z : a.w;
}

// register blocking: one thread computes TP consecutive pixels of a row x TK output channels
// per (row r, 4 input channels): TP + 2 input float4 and 3 * 4 * TK / 4 weight float4 feed 3 * 4 * TP * TK FMAs
// NHWC activations, RSCK weights, clamp padding, bias, ReLU
template <int C, int K, int TP, int TK>
__global__ void conv3x3_blocked(const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ b,
                                float* __restrict__ out, int W, int H)
{
    const int groups = K / TK;                       // threads per pixel tile, one per channel group
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tiles = (W + TP - 1) / TP;
    if (i >= tiles * H * groups) return;
    int k0 = (i % groups) * TK;                      // channel group fastest: neighbouring threads read neighbouring weights
    int t = i / groups;
    int u0 = (t % tiles) * TP, v = t / tiles;

    float acc[TP][TK];
    #pragma unroll
    for (int j = 0; j < TP; ++j)
        #pragma unroll
        for (int k = 0; k < TK; ++k)
            acc[j][k] = b[k0 + k];

    #pragma unroll 1
    for (int r = 0; r < 3; ++r)
    {
        const float* row = in + size_t(min(max(v + r - 1, 0), H - 1)) * W * C;
        int ux[TP + 2];                              // clamped columns u0 - 1 .. u0 + TP
        #pragma unroll
        for (int j = 0; j < TP + 2; ++j)
            ux[j] = min(max(u0 + j - 1, 0), W - 1) * C;

        #pragma unroll 2
        for (int c = 0; c < C; c += 4)
        {
            float4 x[TP + 2];                        // 4 channels of the TP + 2 input pixels, reused by all 3 taps s
            #pragma unroll
            for (int j = 0; j < TP + 2; ++j)
                x[j] = *reinterpret_cast<const float4*>(row + ux[j] + c);

            #pragma unroll
            for (int s = 0; s < 3; ++s)
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc)
                {
                    const float* wp = w + ((r * 3 + s) * C + c + cc) * K + k0;
                    float wk[TK];
                    #pragma unroll
                    for (int k = 0; k < TK; k += 4)
                    {
                        float4 q = *reinterpret_cast<const float4*>(wp + k);
                        wk[k] = q.x; wk[k + 1] = q.y; wk[k + 2] = q.z; wk[k + 3] = q.w;
                    }
                    #pragma unroll
                    for (int j = 0; j < TP; ++j)
                    {
                        float xv = lane(x[j + s], cc);
                        #pragma unroll
                        for (int k = 0; k < TK; ++k)
                            acc[j][k] += xv * wk[k];
                    }
                }
        }
    }

    #pragma unroll
    for (int j = 0; j < TP; ++j)
    {
        if (u0 + j >= W) break;
        float* o = out + (size_t(v) * W + u0 + j) * K + k0;
        #pragma unroll
        for (int k = 0; k < TK; k += 4)
            *reinterpret_cast<float4*>(o + k) = make_float4(fmaxf(acc[j][k], 0.0f), fmaxf(acc[j][k + 1], 0.0f),
                                                            fmaxf(acc[j][k + 2], 0.0f), fmaxf(acc[j][k + 3], 0.0f));
    }
}
