# 01 — CUDA filtering

**Track:** PC series (M1) · **Status:** In progress

GPU implementation of image filters in CUDA, verified against a CPU float64 reference.

| Part | Content | Status |
|---|---|---|
| `HelloCuda` | Toolchain check: upload, kernel, readback | Done |
| `GaussianFilter` | Direct and separable 2D Gaussian, compared with the float64 reference | Done |

```bash
python ../00-image-baseline/run.py               # generates the float64 reference (.npy)
# then build and run CudaFiltering/CudaFiltering.sln (Visual Studio 2022, CUDA build customisation)
```

## Execution model

<a href="figures/cuda_indexing.png"><img src="figures/cuda_indexing.png" width="640"></a>

Same model in HLSL:

<a href="figures/SV_DispatchThreadID.png"><img src="figures/SV_DispatchThreadID.png" width="420"></a>

<sub>Figure: Microsoft, [SV_DispatchThreadID](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/sv-dispatchthreadid), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).</sub>

## DX12 / HLSL → CUDA

| DX12 / HLSL | CUDA |
|---|---|
| Compute shader entry point | `__global__ void kernel(...)` |
| `[numthreads(X,Y,Z)]` | Block size, chosen at launch |
| `Dispatch(X,Y,Z)` | `kernel<<<grid, block>>>(args)` |
| `SV_GroupID`, `SV_GroupThreadID` | `blockIdx`, `threadIdx` |
| `SV_DispatchThreadID` | `blockIdx * blockDim + threadIdx` |
| `CreateCommittedResource` | `cudaMalloc` |
| Descriptor, root signature, PSO | Raw pointer as kernel argument |
| Upload / readback heap + `CopyBufferRegion` | `cudaMemcpy` |
| `groupshared` | `__shared__` |
| Fence wait | `cudaDeviceSynchronize`, blocking `cudaMemcpy` |
| `ExecuteCommandLists` (asynchronous) | Kernel launch (asynchronous) |
| D3D12 debug layer | `compute-sanitizer` (wraps the executable) |
| DXIL → driver → native ISA | PTX → driver or `nvcc` → SASS |

## HelloCuda

```mermaid
flowchart LR
    H["host array<br/>41.0 × n"] -->|cudaMemcpy| D["device buffer"]
    D --> K["add_one&lt;&lt;&lt;4, 256&gt;&gt;&gt;<br/>data[i] += 1"]
    K -->|"cudaMemcpy<br/>(waits for kernel)"| R["host array<br/>42.0 × n"]
```

$$i = \text{blockIdx.x} \cdot \text{blockDim.x} + \text{threadIdx.x}, \qquad \text{grid} = \lceil n / \text{block} \rceil$$

Threads with $i \ge n$ exit without writing (the last block may extend past $n$).

### Correctness

Laptop GPU, $n = 1024$, block 256.

| Check | Result |
|---|---|
| `h[0]`, `h[n-1]` after readback | 42.0, 42.0 |
| `compute-sanitizer` | 0 errors |
| Injected out-of-bounds access ($n$ = 1024, 1000-element buffer) | Normal run: `cudaSuccess`. Sanitizer: invalid read at block 3, thread 232 ($i = 1000$) |

## GaussianFilter

Filter definition, boundary handling and test data: [00 — Image baseline](../00-image-baseline/README.md).

$$\hat{x}[v,u] = \sum_{j=-R}^{R}\sum_{i=-R}^{R} w_j w_i \cdot y\big[c_v(v+j), c_u(u+i)\big], \qquad R = \lceil 3\sigma \rceil$$

Inputs: noisy images ($\sigma_n = 0.1$), 256×256 and 257×129. Non-square size exposes swapped-axis errors.

| Pattern | Clean | Noisy | Exposes |
|---|---|---|---|
| `step_edge` | <img src="../00-image-baseline/figures/noise/step_edge.png" width="96"> | <img src="../00-image-baseline/figures/filters/step_edge_n0.1.png" width="96"> | Kernel shift or asymmetry, border handling |
| `zone_plate` | <img src="../00-image-baseline/figures/noise/zone_plate.png" width="96"> | <img src="../00-image-baseline/figures/filters/zone_plate_n0.1.png" width="96"> | Wrong weights, swapped axes (all frequencies, both axes) |

### Direct

```mermaid
flowchart TD
    subgraph S1["CPU: prepare"]
        direction LR
        A["compute weights w<br/>(float64, normalised)"] --> B["load y, reference<br/>(.npy)"] --> C["narrow y, w<br/>to float32"]
    end
    subgraph S2["Upload and launch"]
        direction LR
        D["cudaMalloc<br/>in, out, w"] --> E["cudaMemcpy<br/>y, w → GPU"] --> F["launch gaussian_direct<br/>grid × 32×32 block"]
    end
    subgraph S3["Execute and verify"]
        direction LR
        G["GPU: one thread<br/>per pixel"] --> H["cudaMemcpy out → CPU<br/>(waits for the kernel)"] --> I["CPU: max error<br/>against reference"]
    end
    S1 --> S2 --> S3
```

Kernel, per thread:

```
(u, v) = blockIdx.xy * blockDim.xy + threadIdx.xy
if u >= W or v >= H: return
sum = 0
for j in [-R, R]:
    for i in [-R, R]:
        sum += w[j+R] * w[i+R] * in[clamp(v+j, 0, H-1) * W + clamp(u+i, 0, W-1)]
out[v * W + u] = sum
```

### Separable

Definition and cost: [00 — Filters](../00-image-baseline/README.md#filters). Inputs, upload and readback as for the direct filter, with one additional buffer `d_tmp`.

```mermaid
flowchart LR
    I["d_in"] --> H["blur_h"] --> T["d_tmp"] --> V["blur_v"] --> O["d_out"]
    W["d_w"] --> H
    W --> V
```

Kernels, per thread:

```
blur_h: out[v * W + u] = sum over i in [-R, R] of w[i+R] * in[v * W + clamp(u+i, 0, W-1)]
blur_v: out[v * W + u] = sum over j in [-R, R] of w[j+R] * in[clamp(v+j, 0, H-1) * W + u]
```

Both launches use the default stream, which executes in order: `blur_v` observes the writes of `blur_h` without an explicit barrier.

### Results

`max |CUDA (float32) − reference (float64)|`, laptop GPU, Release, 16 cases per filter.

| Filter | Pattern, size | $\sigma = 0.5$ | $\sigma = 1$ | $\sigma = 2$ | $\sigma = 4$ |
|---|---|---|---|---|---|
| Direct | Taps per pixel | 25 | 49 | 169 | 625 |
| | `step_edge`, 256×256 | 4.80e-7 | 5.67e-7 | 1.03e-6 | 1.52e-6 |
| | `step_edge`, 257×129 | 4.67e-7 | 5.04e-7 | 8.50e-7 | 1.50e-6 |
| | `zone_plate`, 256×256 | 3.46e-7 | 4.33e-7 | 6.36e-7 | 1.19e-6 |
| | `zone_plate`, 257×129 | 3.05e-7 | 4.63e-7 | 6.21e-7 | 7.99e-7 |
| Separable | Taps per pixel | 10 | 14 | 26 | 50 |
| | `step_edge`, 256×256 | 2.77e-7 | 2.76e-7 | 2.91e-7 | 3.35e-7 |
| | `step_edge`, 257×129 | 2.91e-7 | 3.05e-7 | 2.66e-7 | 4.47e-7 |
| | `zone_plate`, 256×256 | 2.37e-7 | 2.11e-7 | 1.79e-7 | 2.39e-7 |
| | `zone_plate`, 257×129 | 2.11e-7 | 1.80e-7 | 1.68e-7 | 2.18e-7 |

Largest error of each column:

| Quantity | $\sigma = 0.5$ | $\sigma = 1$ | $\sigma = 2$ | $\sigma = 4$ |
|---|---|---|---|---|
| Direct | 4.80e-7 | 5.67e-7 | 1.03e-6 | 1.52e-6 |
| Separable | 2.91e-7 | 3.05e-7 | 2.91e-7 | 4.47e-7 |
| Error ratio, direct / separable | 1.6 | 1.9 | 3.5 | 3.4 |
| √(direct taps / separable taps) | 1.6 | 1.9 | 2.5 | 3.5 |
| `max \|direct − separable\|` | 5.96e-7 | 5.96e-7 | 1.07e-6 | 1.55e-6 |

### Discussion

| Observation | Interpretation |
|---|---|
| Maximum error 1.5e-6 (direct), 4.5e-7 (separable) | float32 rounding ($\varepsilon = 1.2 \times 10^{-7}$ at 1.0): both filters match the reference, including borders and the non-square size |
| Separable error is lower in all 16 cases | Each float32 operation rounds; separable performs fewer operations per pixel (50 terms with one multiplication each, against 625 with two, at $\sigma = 4$) |
| Direct error grows with $\sigma$ | More taps accumulate more rounding error |
| Error ratio follows √(tap ratio), except at $\sigma = 2$ | Consistent with rounding errors of random sign accumulating as a random walk |
| Separable error remains nearly constant with $\sigma$ | Not explained by the tap count. Attenuation of first-pass rounding by the second pass is a candidate; not verified |
| Direct and separable differ by up to 1.55e-6, against 3.1e-15 in float64 | Identical in exact arithmetic; the difference is float32 rounding, dominated by the direct filter |
| Debug and Release differ in the last digits (e.g. 5.38e-7 and 4.80e-7) | Compiled floating-point operations differ; the error magnitude does not |

The separable form requires fewer taps and, in float32, is the more precise evaluation.

## Environment

NVIDIA GeForce RTX 5090 Laptop GPU (compute capability 12.0, `sm_120`), driver 617.14, CUDA 13.4, Visual Studio 2022, Windows 11.
