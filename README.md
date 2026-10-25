# OmdCompute

A personal research lab exploring the boundary between real-time rendering and GPU computing: CUDA kernels and performance analysis, learned image reconstruction, on-device GPU/NPU inference on mobile hardware, and (optionally) ML compilation.

Each experiment is a bounded, self-contained investigation with a stated question, a baseline, correctness checks and measured results on real hardware. See [ROADMAP.md](ROADMAP.md) for the full plan.

## Status

### PC series

| # | Experiment | Status |
|---|---|---|
| 00 | [Image baseline](experiments/pc/00-image-baseline/) — CPU reference filters and metrics | In progress |
| 01 | [CUDA filtering](experiments/pc/01-cuda-filtering/) — correctness and profiler-guided optimisation | Not started |
| 02 | [Neural denoising](experiments/pc/02-neural-denoising/) — compact residual CNN in PyTorch | Not started |
| 03 | [Reconstruction](experiments/pc/03-reconstruction/) — temporal reconstruction or learned upscaling | Not started |
| 04 | [ML compilation](experiments/pc/04-ml-compilation/) — optional | Not started |

### Mobile menu (independent, chosen one at a time)

| ID | Experiment | Status |
|---|---|---|
| G1 | [Vulkan filter comparison](experiments/mobile/g1-vulkan-filters/) | Not started |
| G2 | [Upsampling viewer](experiments/mobile/g2-upsampling-viewer/) | Not started |
| G3 | [Tiny volumetric scene](experiments/mobile/g3-volumetric-scene/) | Not started |
| G4 | [Bandwidth/precision study](experiments/mobile/g4-bandwidth-precision/) | Not started |
| N1 | [Inference backend inspector](experiments/mobile/n1-inference-inspector/) | Not started |
| N2 | [Quantization explorer](experiments/mobile/n2-quantization-explorer/) | Not started |
| N3 | [Neural enhancement viewer](experiments/mobile/n3-neural-enhancement/) | Not started |
| N4 | [Depth-guided graphics effect](experiments/mobile/n4-depth-guided-effect/) | Not started |

## Layout

```
docs/            context, learning notes, decision records, report template
experiments/     pc/ (numbered series) and mobile/ (independent menu)
fixtures/        small shared test images or their generation recipes
reports/         selected images, tables and conclusions
```
