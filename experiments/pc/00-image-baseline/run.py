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

# test zone-plate
from fixtures import PATTERNS
from preview import save_png

# baseline patterns
for W, H in cfg["sizes"]:
    for name, params in cfg["patterns"].items():
        x = PATTERNS[name](W, H, **params)
        save_png(out_dir / f"{name}_{W}x{H}.png", x)


from noise import unit_noise, add_noise

# clean pattern + noised image
for W, H in cfg["sizes"]:
    z = unit_noise(W, H, cfg["seed"])
    print(f"{W}x{H}: z mean {z.mean():+.4f}, std {z.std():.4f}")   # sanity check
    for name, params in cfg["patterns"].items():
        x = PATTERNS[name](W, H, **params)
        # save the baseline pattern
        save_png(out_dir / f"{name}_{W}x{H}.png", x)
        for s in cfg["noise"]["sigmas"]:
            y = add_noise(x, z, s)
            # save the noisy image
            save_png(out_dir / f"{name}_{W}x{H}_n{s}.png", y)