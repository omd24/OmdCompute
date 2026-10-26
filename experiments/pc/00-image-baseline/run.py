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
from preview import save_png, save_error_png
from noise import unit_noise, add_noise
from filters import filter_variants
import csv
from metrics import mse, psnr

# main loop
# generate clean patterns -> apply noise on the pattern -> filter the noised image
variants = filter_variants(cfg["filters"])     # extract filter info
rows = [] 
for W, H in cfg["sizes"]:
    z = unit_noise(W, H, cfg["seed"])
    print(f"{W}x{H}: z mean {z.mean():+.4f}, std {z.std():.4f}")   # sanity check
    for name, params in cfg["patterns"].items():
        # 1. call function ptr to generate pattern:
        x = PATTERNS[name](W, H, **params)
        img_dir = out_dir / "images" / name
        img_dir.mkdir(parents=True, exist_ok=True)
        save_png(img_dir / f"{W}x{H}.png", x) # save the baseline pattern
        for s in cfg["noise"]["sigmas"]:
            # 2. apply gaussian noise:
            y = add_noise(x, z, s)
            save_png(img_dir / f"{W}x{H}_n{s}.png", y) # save the noisy image
            for f_name, f_fn, f_params in variants:
                # 3. filter the noisy image
                x_hat = f_fn(y, **f_params)
                save_png(img_dir / f"{W}x{H}_n{s}_{f_name}.png", x_hat) # save the filtered result
                rows.append({"pattern": name, "size": f"{W}x{H}", "sigma_n": s,
                             "filter": f_name, "mse": mse(x_hat, x),
                             "psnr": psnr(x_hat, x, cfg["metrics"]["peak"])})
                save_error_png(img_dir / f"{W}x{H}_n{s}_{f_name}_err.png", x_hat - x, s)

# test 1D gaussian weights
if False:
    from filters import gaussian_kernel_1d
    test_weights = gaussian_kernel_1d(1.0, 3.0); # sigma = 1, truncate at 3*sigma = 3
    print(test_weights)

# write metrics into csv
with open(out_dir / "metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
