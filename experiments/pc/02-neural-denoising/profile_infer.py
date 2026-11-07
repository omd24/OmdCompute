"""Runs warm-up inference passes, then marks one 2048x2048 forward pass as the capture range for Nsight."""
import json
import time
from pathlib import Path

import torch

from model import Denoiser

ROOT = Path(__file__).parent
SIZE = 2048
WARMUP = 3


@torch.no_grad()
def main(run="depth13"):
    cfg = json.loads((ROOT / "runs" / run / "config.json").read_text())
    model = Denoiser(cfg["depth"], cfg["features"]).cuda().eval()
    model.load_state_dict(torch.load(ROOT / "runs" / run / "model.pt"))
    y = torch.rand(1, 1, SIZE, SIZE, device="cuda")  # content does not change the cost

    for _ in range(WARMUP):  # cuDNN algorithm choice and allocator warm-up
        model(y)
    torch.cuda.synchronize()

    torch.cuda.profiler.start()  # ncu --profile-from-start off records only from here
    start = time.perf_counter()
    model(y)
    torch.cuda.synchronize()
    torch.cuda.profiler.stop()
    print(f"{SIZE}x{SIZE}  {(time.perf_counter() - start) * 1e3:.1f} ms for the captured pass")


if __name__ == "__main__":
    main()