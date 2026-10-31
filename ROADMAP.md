# OmdCompute — Roadmap

## 1. Purpose

The lab takes real-time rendering as its starting point and extends into adjacent fields: general GPU computing, modern ML engineering, neural graphics and computer vision, on-device inference, and optionally ML compilation.

It is a long-term learning lab rather than a single demo. The emphasis is on fundamentals, current engineering practice and selected recent research — not on invoking libraries. Each milestone is intended to stand on its own as a documented result.

## 2. Learning areas

| Area | Fundamentals | Tools / workflow | Connection to rendering |
|---|---|---|---|
| CUDA and GPU computing | Execution model, memory hierarchy, parallel algorithms, occupancy, bandwidth, synchronisation, numerical behaviour | CUDA C++, Nsight Systems/Compute, repeatable benchmarks | Extends shader and GPU-performance knowledge to general kernels |
| ML engineering | Tensors, autodiff, loss design, optimisation, generalisation, inference, reproducibility | Python, PyTorch, data pipelines, configs, checkpoints, profiling | Learned reconstruction as a complement to hand-tuned filters |
| Computer vision / neural graphics | Image formation, filtering, restoration, denoising, super-resolution, depth, temporal reconstruction | Datasets, evaluation scripts, model export, visualisation | Reuses image and sampling mathematics; rendered versus real-camera data |
| Efficient / on-device ML | Quantisation, precision, conversion, operator support, partitioning, cold vs. steady-state cost | LiteRT, QNN or another verified runtime, backend diagnostics | Hardware-aware quality/performance trade-offs |
| Mobile graphics / systems | Shared memory, mobile GPU constraints, Vulkan compute, thermal and sustained performance | Android Studio, SDK/NDK, Gradle/CMake, ADB, profiling | Desktop/console knowledge applied to power- and bandwidth-limited hardware |
| Compilers and ML systems | Graph/IR, lowering, fusion, specialisation, floating-point semantics | `torch.compile` diagnostics first; optional LLVM/MLIR or Triton | Shader compilation and GPU scheduling compared with model compilation |
| Research literacy | Assumptions, baselines, limited reproduction, failure cases | Primary papers, official documentation, short reports | Graphics research habits applied to neighbouring fields |

Only one topic is active at a time, following the cycle: **fundamental → modern technique → bounded experiment → measurement → discussion of what changed and what remains unclear.**

Recent work is reviewed, with dated primary sources, when each extension is chosen; no architecture is fixed in advance. Large language models are out of scope for the initial phase.

## 3. Hardware and environment

| Target | Description |
|---|---|
| Desktop | Windows workstation with an NVIDIA CUDA-capable GPU |
| Android phone | Arm SoC with an integrated mobile GPU and an NPU |

Exact hardware and software versions are recorded in the environment section of each experiment's report. Phone capabilities are queried at runtime before an implementation is committed to.

Phone memory is shared between CPU and GPU and is not all available to one application; shared memory does not remove API synchronisation or guarantee zero-copy sharing. The phone is not a CUDA target.

### Workflow

- **Discussion and implementation** both take place in Claude Code sessions opened in this repository, so theory discussions can refer to the actual code, logs and images. Every discussion ends with a written outcome (learning note, decision record or README update).
- **Laptop sessions** (desk or phone via Remote Control) are the default, as nearly all work needs the CUDA GPU. The laptop must be running and awake.
- **Cloud sessions** are limited to CPU-only work: documentation, CPU references, tests. They have no access to the CUDA GPU or the phone.
- **Phone deployment** uses ADB wireless debugging from the laptop: APKs are installed, launched and their logs collected by the laptop session. Mobile GPU/NPU results come only from the installed app.
- **Linux host:** CUDA and PyTorch are used natively on Windows. A WSL2 Ubuntu environment is added if and when the Qualcomm/LiteRT NPU toolchain requires it.

## 4. Structure

The PC series carries the substantial work. Mobile is a menu of independently finishable experiments, not a port of the PC lab; both share fixtures, metrics and reporting conventions.

| Track | Experiment | Outcome |
|---|---|---|
| PC 00 | Image baseline | Reproducible CPU reference images and metrics |
| PC 01 | CUDA filtering | Correct CUDA variants, benchmarks and profiler-backed reasoning |
| PC 02 | Neural denoising | Training/evaluation workflow and quality-versus-cost comparison |
| PC 03 | Reconstruction | Temporal reconstruction **or** learned upscaling, using own renderer data where appropriate |
| PC 04 | ML compilation (optional) | Compilation-behaviour report; optional limited IR transformation |
| Mobile GPU | One of G1–G4 | Small visual/performance result on the mobile GPU |
| Mobile NPU | N1, then optionally N2–N4 | Verified deployment and a small vision application |

Recommended order: **PC 00**, with **G1 or N1** as an independent side exercise. PC 03 and PC 04 are alternatives after a useful PC 02 result, not simultaneous obligations.

## 5. PC milestones

### 00 — Image baseline

- A few float32 clean/noisy images with fixed seeds: gradients, edges, checkerboards, flat fields.
- Linear colour, intensity range, channel order and boundary behaviour are specified.
- Identity and CPU Gaussian references; an edge-preserving filter only after the simple baseline works.
- Previews, configs and MSE/PSNR are saved; zero-error PSNR and small/non-square inputs are handled.

**Done:** one command reproduces a small image/metrics report. Gaussian toy noise is labelled as such, not as Monte Carlo rendering noise.

### 01 — CUDA filtering

- Mathematically equivalent CUDA filters, compared with the CPU reference within justified tolerances.
- A simple kernel first; profiling precedes tiling, shared memory or other changes.
- Warmed GPU execution is measured with synchronisation, separately from transfers, allocation and end-to-end latency.
- Image sizes, access patterns, separability and launch configuration are examined; successful and unsuccessful optimisations are both explained.

**Done:** correct outputs, a repeatable comparison and a short profiler-backed explanation on the desktop GPU, with hardware/software versions recorded.

### 02 — Neural denoising

- A modest residual CNN in PyTorch, with explicit study of what the network, loss and optimiser do.
- Comparison against the noisy input and classical filters on held-out data.
- Training config, curves, model size, inference time and representative failures are saved.
- Data is split by scene/sequence to avoid leakage between near-identical frames.
- Optionally, the CUDA filter is exposed as a PyTorch custom operator once standalone correctness is established.

**Done:** reproducible training and evaluation with a clearly stated domain limitation. Toy-data success is not evidence of general rendering-denoising quality.

### 03 — Temporal reconstruction or upscaling

- The existing personal renderer is inspected before deciding how to export colour, motion, depth or normals.
- Input/reference pairs are defined defensibly; no physically based sampling-noise claims without a stochastic estimator and reference.
- One focus is selected: temporal denoising, learned upscaling or a related reconstruction problem.
- Reprojection, history rejection, disocclusion and temporal stability are studied, starting with one bounded idea.
- Evaluation covers short sequences and failure cases, not only still-image PSNR.

**Done:** a limited but substantial experiment with visual evidence, performance numbers and an analysis of failures. No third-party proprietary data, code or assets.

### 04 — ML compilation (optional)

- The same model compared in eager and `torch.compile` modes.
- Compilation/first-call cost separated from steady state; graph breaks, specialisation and generated code inspected.
- Observations related to fusion, launch overhead and hardware constraints; compilation is not assumed to help.
- Optional deeper step: one limited LLVM/MLIR transformation with before/after IR, semantic tests and a stated validity domain. Floating-point reassociation is not assumed exact.

**Done:** a compilation-behaviour report and, if implemented, a small transformation with correctness evidence.

## 6. Mobile menu

One experiment at a time. Build and deployment plumbing may take longer than the algorithm itself. Bundled still images come first; camera and live video are extensions. A shared app shell is introduced only after a first experiment works.

| ID | Idea | First scope | Evidence |
|---|---|---|---|
| G1 | GPU filter comparison | CPU reference, naive Vulkan blur, then separable | Workgroup choices, numerical checks, warmed GPU and end-to-end timings |
| G2 | Upsampling viewer | Bilinear vs. bicubic; Lanczos later | Ringing/sharpness/cost; zoomed crops and reference metrics |
| G3 | Tiny volumetric scene | One light, simple fog, limited raymarching | Step count, reduced resolution, precision; visual trade-offs and GPU time |
| G4 | Bandwidth/precision study | Read-transform-write with several formats | Storage/arithmetic precision, synchronisation, sustained behaviour |
| N1 | Inference backend inspector | One small pretrained vision model; CPU, GPU, NPU added separately | Layouts, conversion, actual partitioning; output checks, cold/warm timing, visible fallback |
| N2 | Quantisation explorer | Float and quantised exports of the N1 model | Calibration, accuracy, operator support; quality and cost |
| N3 | Neural enhancement viewer | Compact pretrained denoiser or SR model on stills | Deployment, tiling/seams; classical baseline and stage latency |
| N4 | Depth-guided graphics effect | Small monocular-depth model, then GPU fog/stylisation | NPU/GPU handoff, pre/post-processing, stage timings |

**First GPU choice: G1. First NPU choice: N1.** Pretrained models require licence and export checks before distribution. Monocular depth is an estimate, not ground-truth geometry.

### NPU feasibility gate

The NPU is reached through a supported runtime/compiler (e.g. LiteRT with Qualcomm AI Engine Direct / QNN), not through hand-written kernels. Before committing to NPU work:

1. Android/firmware, runtime and vendor library versions are recorded, and the required conversion host is verified.
2. A known small supported model with fixed input shape is chosen; operators, layout and precision are checked.
3. CPU inference is demonstrated first; acceleration is added separately.
4. Logs/profiling distinguish full NPU execution, partial delegation and CPU fallback. Inconclusive evidence is reported as such.
5. Initialisation, pre-processing, inference, post-processing and end-to-end latency are measured.

**Selecting the NPU delegate is not proof of NPU execution.** Mobile timings record resolution, device state, charging/power mode, warm-up and run duration; short bursts are separated from sustained runs. Energy efficiency is not inferred from battery percentage or temperature.

## 7. Experiment loop

1. **Discuss:** one question, the relevant theory, a baseline and what would count as an informative result.
2. **Brief:** algorithm, scope, execution target, inputs, checks and expected report.
3. **Implement:** minimal change; run whatever the current environment genuinely supports.
4. **Run on target hardware:** laptop for CUDA work, installed app for mobile GPU/NPU work. Unexecuted tests are labelled.
5. **Interpret:** relate results to theory and choose the next step.
6. **Checkpoint:** experiment README, learning notes, decision records and the status table are updated.

Every result is written up using [docs/report-template.md](docs/report-template.md).

## 8. Resources

Resources are consulted when a milestone needs them, starting from official documentation and primary papers. For each research question, at most one foundational and one recent source are chosen initially.

| Resource | Purpose |
|---|---|
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html) | Execution/memory model |
| [PyTorch tutorials](https://docs.pytorch.org/tutorials/) | Training, inference, profiling |
| [Custom C++/CUDA operators](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html) | Optional integration |
| [DnCNN](https://arxiv.org/abs/1608.03981) | Residual denoising pattern |
| [SVGF](https://research.nvidia.com/labs/rtr/publication/schied2017spatiotemporal/) | Spatiotemporal variance-guided filtering |
| [torch.compile introduction](https://docs.pytorch.org/tutorials/intermediate/torch_compile_tutorial) | Model compilation |
| [torch.compile end-to-end](https://docs.pytorch.org/tutorials/intermediate/torch_compile_full_example.html) | Compile overhead vs. steady state |
| [LLVM Kaleidoscope](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/) | Compiler pipeline fundamentals |
| [MLIR Toy](https://mlir.llvm.org/docs/Tutorials/Toy/) | IR, transformations, lowering |
| [Android Studio](https://developer.android.com/studio/install) | Host tooling |
| [Android game samples](https://developer.android.com/games/samples) | Native/Vulkan starting points |
| [LiteRT overview](https://developers.google.com/edge/litert/overview) | Conversion and runtime |
| [LiteRT NPU](https://developers.google.com/edge/litert/next/npu) | NPU compilation/deployment |
| [LiteRT Qualcomm](https://developers.google.com/edge/litert/next/qualcomm) | Supported SoCs, host requirements |
| [Qualcomm AI Hub](https://aihub.qualcomm.com/) | Reference models (hosted results are not measurements of this device) |
| [Claude Code Remote Control](https://code.claude.com/docs/en/remote-control) | Phone-directed laptop sessions |
| [Claude Code on the web](https://code.claude.com/docs/en/claude-code-on-the-web) | Cloud (CPU-only) sessions |

## 9. Repository policy

One repository with independent experiments. Large datasets, checkpoints and build outputs are excluded from Git, with reproducible download/generation steps and licences documented. CI, if added, runs only checks a hosted CPU runner can validate. An experiment is split into its own repository only when it becomes a standalone application/library, or needs independent releases.
