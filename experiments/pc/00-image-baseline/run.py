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

for W, H in cfg["sizes"]:
    for name, params in cfg["patterns"].items():
        x = PATTERNS[name](W, H, **params)
        save_png(out_dir / f"{name}_{W}x{H}.png", x)