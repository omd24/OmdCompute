# OmdCompute — agent conventions

A long-term lab for GPU computing (CUDA), neural graphics / computer vision, and on-device GPU/NPU deployment, with an optional ML-compilation branch. The full plan is in `ROADMAP.md`.

## Execution rules

- Execution targets are distinct: **laptop** (CUDA GPU: kernels, training, profiling), **cloud session** (CPU only, no GPU, no phone), **Android phone** (mobile GPU / NPU, only via an installed app). State which one produced each result.
- Anything not actually executed is labelled as not executed. Benchmark values, backend usage (e.g. "ran on the NPU"), papers and success claims are never invented or inferred.
- Timings separate warm-up/compilation, transfers, and steady-state execution; hardware and software versions are recorded with every result.
- Tests check meaningful properties or agreement with a reference, not just the implementation's own assumptions.

## Repository conventions

- Each experiment under `experiments/` is self-contained: its own README, build/config, tests and results summary. No shared framework until duplication is a real problem; building a mobile experiment must not require CUDA, and vice versa.
- Results follow `docs/report-template.md`. Reports are kept concise; full logs live alongside.
- Large datasets, checkpoints and build outputs stay out of Git; document how to regenerate or download them, including licenses.
- No proprietary or confidential third-party data, code or assets.
- Design discussions end with a written outcome: theory in `docs/learning-notes/`, choices in `docs/decisions/`, status in the experiment README and the status table in `README.md`. A new session should never need chat history to continue.