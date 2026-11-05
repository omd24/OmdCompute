"""Trains the denoiser on DIV2K patches. Writes config, loss curve and weights to runs/<name>/."""
## loops over batches, updates weights, logs curves, saves the model.

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset import PatchDataset, add_noise
from model import Denoiser

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "gray"
RUNS = ROOT / "runs"

# 13 layer network with learning rate of 0.001:  newWeight = oldWeight - lr * step 
# 400 batches of size 32 (each batch is 32 patches and each patch is 64x64 pixels) to be trained per epoch (for 100 epochs in total)

CONFIG = dict(name="depth13", depth=13, features=64, sigma=0.1, patch=64,
              patches_per_image=16, batch=32, epochs=100, lr=1e-3,
              valid_crop=256, seed=0)


def validation_set(cfg):
    # fixed centre crops with fixed noise, so the curve compares like with like
    crops = []
    for f in sorted((DATA / "valid").glob("*.npy")):
        image = np.load(f)
        h, w = image.shape
        c = cfg["valid_crop"]
        y, x = (h - c) // 2, (w - c) // 2
        crops.append(image[y:y + c, x:x + c])
    x = torch.from_numpy(np.stack(crops))[:, None].cuda()
    g = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    y = x + cfg["sigma"] * torch.randn(x.shape, generator=g, device="cuda")
    return x, y


@torch.no_grad()
def validate(model, x, y):
    model.eval()
    mse = sum(F.mse_loss(model(y[i:i + 25]), x[i:i + 25], reduction="sum")
              for i in range(0, len(x), 25)) / x.numel()
    model.train()
    return mse.item()


def main(cfg=CONFIG):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    out = RUNS / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    loader = DataLoader(PatchDataset(DATA / "train", cfg["patch"], cfg["patches_per_image"]),
                        batch_size=cfg["batch"], shuffle=True, drop_last=True)
    valid_x, valid_y = validation_set(cfg)
    model = Denoiser(cfg["depth"], cfg["features"]).cuda()
    optimiser = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    noisy = F.mse_loss(valid_y, valid_x).item()
    print(f"validation input MSE {noisy:.6f}")
    with open(out / "curve.csv", "w") as curve:
        curve.write("epoch,train_mse,valid_mse,seconds\n")
        for epoch in range(cfg["epochs"]):
            start = time.perf_counter()
            total = torch.zeros((), device="cuda")  # summed on the GPU, read once per epoch
            for x in loader:
                x = x.cuda(non_blocking=True)                           # copy batch from cpu to gpu. non_blocking would do async copy if memory was pinned (page-locked)
                loss = F.mse_loss(model(add_noise(x, cfg["sigma"])), x) # MSE between denoised noisy patch and clean patch
                optimiser.zero_grad()                                   # clear old gradients                
                loss.backward()                                         # compute new gradients
                optimiser.step()                                        # update weights
                total += loss.detach()                                  # add the loss to running sum (on gpu). detach() drops gradient history, so memory won't grow
            train = total.item() / len(loader)
            valid = validate(model, valid_x, valid_y)
            seconds = time.perf_counter() - start
            curve.write(f"{epoch},{train:.6f},{valid:.6f},{seconds:.2f}\n")
            print(f"epoch {epoch:3d}  train {train:.6f}  valid {valid:.6f}  {seconds:.1f} s")
    torch.save(model.state_dict(), out / "model.pt")


if __name__ == "__main__":
    main()