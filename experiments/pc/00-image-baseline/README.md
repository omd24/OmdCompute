# 00 — Image baseline

**Track:** PC series (M0) · **Status:** In progress

Procedural test images with exact ground truth, used as the reference for the filtering and denoising experiments.

```bash
python run.py --config configs/default.json
```

## Test patterns

$u \in [0, W-1]$: column, $v \in [0, H-1]$: row. Values are linear intensities in $[0, 1]$.

| Pattern | Definition | Preview |
|---|---|---|
| `gradient` | $x = \dfrac{u}{W-1}$ | <img src="figures/patterns/gradient.png" width="96"> |
| `diagonal_gradient` | $x = \dfrac12\left(\dfrac{u}{W-1} + \dfrac{v}{H-1}\right)$ | <img src="figures/patterns/diagonal_gradient.png" width="96"> |
| `step_edge` | $x = [\,u \ge W/2\,]$ | <img src="figures/patterns/step_edge.png" width="96"> |
| `checkerboard` | $x = \left(\lfloor u/k \rfloor + \lfloor v/k \rfloor\right) \bmod 2$ | <img src="figures/patterns/checkerboard.png" width="96"> |
| `flat` | $x = c$ | <img src="figures/patterns/flat.png" width="96"> |
| `zone_plate` | $x = \tfrac12 + \tfrac12\cos(\alpha r^2)$, $\ \alpha = \pi f_{max} / r_{max}$ | <img src="figures/patterns/zone_plate.png" width="96"> |

Zone plate: $r$ is the distance from the centre $\left(\frac{W-1}{2}, \frac{H-1}{2}\right)$, $r_{max}$ the centre-to-corner distance; local frequency $f(r) = \alpha r / \pi$ cycles/pixel.

## Noise

$$y = x + \sigma_n z, \qquad z[v,u] \sim \mathcal{N}(0, 1) \ \text{i.i.d.}$$

- $z$: one standard-normal field per image size, generator seeded with $(\text{seed}, W, H)$; shared across patterns and $\sigma_n$.
- $y$ is not clipped, so $\mathbb{E}[y - x] = 0$ at every pixel.

| Pattern | Clean | $\sigma_n = 0.05$ | $\sigma_n = 0.2$ |
|---|---|---|---|
| `step_edge` | <img src="figures/noise/step_edge.png" width="96"> | <img src="figures/noise/step_edge_n0.05.png" width="96"> | <img src="figures/noise/step_edge_n0.2.png" width="96"> |
| `zone_plate` | <img src="figures/noise/zone_plate.png" width="96"> | <img src="figures/noise/zone_plate_n0.05.png" width="96"> | <img src="figures/noise/zone_plate_n0.2.png" width="96"> |

**Note:** only the PNG previews are clipped to $[0, 1]$; all computations use the unclipped values.

## Configuration

| Key | Example | Meaning |
|---|---|---|
| `seed` | `1234` | Base seed of the noise generator |
| `sizes` | `[[256, 256], [257, 129]]` | Image sizes $[W, H]$ |
| `patterns` | `{"checkerboard": {"cell": 8}}` | Patterns to generate and their parameters ($k$ = `cell`, $c$ = `value`, $f_{max}$ = `max_frequency`) |
| `noise.sigmas` | `[0.02, 0.05, 0.1, 0.2]` | Noise standard deviations $\sigma_n$ |
| `output_dir` | `"results/default"` | Output folder, relative to this directory |
