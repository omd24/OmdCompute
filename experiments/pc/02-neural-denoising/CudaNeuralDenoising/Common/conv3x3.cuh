#pragma once

#include <cuda_pipeline.h>

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

// shared-memory tiling: one block = PX consecutive pixels of row v x all K channels
// threads: (PX / TP) pixel tiles x (K / TK) channel groups; each thread computes TP pixels x TK channels as in conv3x3_blocked
// per (row r, chunk of CC input channels) the block loads the input segment (+1 apron pixel each side) and the weights once
template <int C, int K, int TP, int TK, int PX, int CC>
__global__ void conv3x3_smem(const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ b,
                             float* __restrict__ out, int W, int H)
{
    __shared__ float4 xs[PX + 2][CC / 4];            // input: pixel in segment, float4 of channels
    __shared__ float4 ws[3][CC][K / 4];              // weights: tap s, channel in chunk, float4 of output channels

    const int groups = K / TK;
    int g = threadIdx.x % groups;                    // channel group: output channels g*TK .. g*TK + TK-1
    int tile = threadIdx.x / groups;                 // pixel tile within the segment
    int ub = blockIdx.x * PX;                        // first pixel of the segment
    int u0 = ub + tile * TP;                         // first pixel of this thread
    int v = blockIdx.y;
    //int k0 = g * TK;

    float acc[TP][TK];
    #pragma unroll
    for (int j = 0; j < TP; ++j)
        #pragma unroll
        for (int k = 0; k < TK; ++k)
            acc[j][k] = b[(k / 4) * 4 * groups + 4 * g + k % 4];

    #pragma unroll 1
    for (int r = 0; r < 3; ++r)
    {
        const float* row = in + size_t(min(max(v + r - 1, 0), H - 1)) * W * C;

        #pragma unroll 1
        for (int c0 = 0; c0 < C; c0 += CC)
        {
            // cooperative loads: consecutive threads read consecutive float4
            for (int i = threadIdx.x; i < (PX + 2) * CC / 4; i += blockDim.x)
            {
                int px = i / (CC / 4), q = i % (CC / 4);
                int col = min(max(ub + px - 1, 0), W - 1);           // px 0 is the left apron pixel
                xs[px][q] = *reinterpret_cast<const float4*>(row + size_t(col) * C + c0 + 4 * q);
            }
            for (int i = threadIdx.x; i < 3 * CC * K / 4; i += blockDim.x)
            {
                int s = i / (CC * K / 4), e = i % (CC * K / 4);      // RSCK: (r, s, c0..c0+CC-1, all K) is contiguous
                (&ws[s][0][0])[e] = *reinterpret_cast<const float4*>(w + ((r * 3 + s) * C + c0) * K + 4 * e);
            }
            __syncthreads();

            // same outer product as conv3x3_blocked, reading shared memory
            #pragma unroll
            for (int c = 0; c < CC; c += 4)
            {
                float4 x[TP + 2];
                #pragma unroll
                for (int j = 0; j < TP + 2; ++j)
                    x[j] = xs[tile * TP + j][c / 4];                 // pixels u0-1 .. u0+TP
                #pragma unroll
                for (int s = 0; s < 3; ++s)
                    #pragma unroll
                    for (int cc = 0; cc < 4; ++cc)
                    {
                        float wk[TK];
                        #pragma unroll
                        for (int k = 0; k < TK; k += 4)
                        {
                            float4 q = ws[s][c + cc][(k / 4) * groups + g];
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
            __syncthreads();                         // all reads done before the next chunk overwrites xs, ws
        }
    }

    #pragma unroll
    for (int j = 0; j < TP; ++j)
    {
        if (u0 + j >= W) break;                      // last segment of a row may extend past the image
        float* o = out + (size_t(v) * W + u0 + j) * K + 4 * g;
        #pragma unroll
        for (int k = 0; k < TK; k += 4)
            *reinterpret_cast<float4*>(o + (k / 4) * 4 * groups) = make_float4(fmaxf(acc[j][k], 0.0f), fmaxf(acc[j][k + 1], 0.0f),
                                                            fmaxf(acc[j][k + 2], 0.0f), fmaxf(acc[j][k + 3], 0.0f));
    }
}

// conv3x3_smem with double-buffered staging: cp.async copies chunk q + 1 from global to shared memory
// while the block computes chunk q; chunk q = (row r, input channels c0 .. c0 + CC - 1), 3 * C / CC chunks
template <int C, int K, int TP, int TK, int PX, int CC>
__global__ void conv3x3_async(const float* __restrict__ in, const float* __restrict__ w, const float* __restrict__ b,
                              float* __restrict__ out, int W, int H)
{
    __shared__ float4 xs[2][PX + 2][CC / 4];         // two buffers: one being computed, one being filled
    __shared__ float4 ws[2][3][CC][K / 4];

    const int groups = K / TK;
    const int chunks = 3 * (C / CC);
    int g = threadIdx.x % groups;
    int tile = threadIdx.x / groups;
    int ub = blockIdx.x * PX;
    int u0 = ub + tile * TP;
    int v = blockIdx.y;

    // issues the asynchronous copies of chunk q into buffer q % 2
    auto stage = [&](int q)
    {
        int r = q / (C / CC), c0 = (q % (C / CC)) * CC, buf = q % 2;
        const float* row = in + size_t(min(max(v + r - 1, 0), H - 1)) * W * C;
        for (int i = threadIdx.x; i < (PX + 2) * CC / 4; i += blockDim.x)
        {
            int px = i / (CC / 4), e = i % (CC / 4);
            int col = min(max(ub + px - 1, 0), W - 1);
            __pipeline_memcpy_async(&xs[buf][px][e], row + size_t(col) * C + c0 + 4 * e, sizeof(float4));
        }
        for (int i = threadIdx.x; i < 3 * CC * K / 4; i += blockDim.x)
        {
            int s = i / (CC * K / 4), e = i % (CC * K / 4);
            __pipeline_memcpy_async(&ws[buf][s][0][0] + e, w + ((r * 3 + s) * C + c0) * K + 4 * e, sizeof(float4));
        }
        __pipeline_commit();
    };

    float acc[TP][TK];
    #pragma unroll
    for (int j = 0; j < TP; ++j)
        #pragma unroll
        for (int k = 0; k < TK; ++k)
            acc[j][k] = b[(k / 4) * 4 * groups + 4 * g + k % 4];

    stage(0);
    #pragma unroll 1
    for (int q = 0; q < chunks; ++q)
    {
        if (q + 1 < chunks)
        {
            stage(q + 1);                            // overlaps with the compute of chunk q
            __pipeline_wait_prior(1);                // chunk q has arrived (this thread's copies)
        }
        else
            __pipeline_wait_prior(0);
        __syncthreads();                             // chunk q has arrived for all threads

        int buf = q % 2;
        #pragma unroll
        for (int c = 0; c < CC; c += 4)
        {
            float4 x[TP + 2];
            #pragma unroll
            for (int j = 0; j < TP + 2; ++j)
                x[j] = xs[buf][tile * TP + j][c / 4];
            #pragma unroll
            for (int s = 0; s < 3; ++s)
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc)
                {
                    float wk[TK];
                    #pragma unroll
                    for (int k = 0; k < TK; k += 4)
                    {
                        float4 t = ws[buf][s][c + cc][(k / 4) * groups + g];
                        wk[k] = t.x; wk[k + 1] = t.y; wk[k + 2] = t.z; wk[k + 3] = t.w;
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
        __syncthreads();                             // buffer q % 2 is refilled by stage(q + 2) in the next iteration
    }

    #pragma unroll
    for (int j = 0; j < TP; ++j)
    {
        if (u0 + j >= W) break;
        float* o = out + (size_t(v) * W + u0 + j) * K + 4 * g;
        #pragma unroll
        for (int k = 0; k < TK; k += 4)
            *reinterpret_cast<float4*>(o + (k / 4) * 4 * groups) = make_float4(fmaxf(acc[j][k], 0.0f), fmaxf(acc[j][k + 1], 0.0f),
                                                                               fmaxf(acc[j][k + 2], 0.0f), fmaxf(acc[j][k + 3], 0.0f));
    }
}
