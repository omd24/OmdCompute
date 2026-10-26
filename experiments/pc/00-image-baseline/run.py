import argparse, json

from pathlib import Path
HERE = Path(__file__).parent

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=HERE / "configs" / "default.json")
args = parser.parse_args()

with open(args.config) as f:
    cfg = json.load(f)

# test json parsed
print(cfg["noise"])

out_dir = HERE / cfg["output_dir"]
out_dir.mkdir(parents=True, exist_ok=True)

from fixtures import PATTERNS
from preview import save_png
from noise import unit_noise, add_noise
from filters import filter_variants

# generate clean patterns -> apply noise on the pattern -> filter the noised image
variants = filter_variants(cfg["filters"])     # extract filter info
for W, H in cfg["sizes"]:
    z = unit_noise(W, H, cfg["seed"])
    print(f"{W}x{H}: z mean {z.mean():+.4f}, std {z.std():.4f}")   # sanity check
    for name, params in cfg["patterns"].items():
        # 1. call function ptr to generate pattern:
        x = PATTERNS[name](W, H, **params)
        save_png(out_dir / f"{name}_{W}x{H}.png", x) # save the baseline pattern
        for s in cfg["noise"]["sigmas"]:
            # 2. apply gaussian noise:
            y = add_noise(x, z, s)
            save_png(out_dir / f"{name}_{W}x{H}_n{s}.png", y) # save the noisy image
            for f_name, f_fn, f_params in variants:
                # 3. filter the noisy image
                x_hat = f_fn(y, **f_params)
                save_png(out_dir / f"{name}_{W}x{H}_n{s}_{f_name}.png", x_hat) # save the filtered result

# test 1D gaussian weights
if False:
    from filters import gaussian_kernel_1d
    test_weights = gaussian_kernel_1d(1.0, 3.0); # sigma = 1, truncate at 3*sigma = 3
    print(test_weights)
