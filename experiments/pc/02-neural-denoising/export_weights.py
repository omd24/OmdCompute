"""Exports a trained denoiser for the CUDA implementation: BatchNorm folded into the convolutions, weights in KRSC order."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model import Denoiser

ROOT = Path(__file__).parent
SIGMA = 0.1
CROP = 256


def fold(model):
    # one (weight KCRS, bias K) pair per conv, with the following BatchNorm folded in
    layers, mods = [], list(model.noise)
    for i, m in enumerate(mods):
        if not isinstance(m, torch.nn.Conv2d):
            continue
        w = m.weight.detach().double()
        b = m.bias.detach().double() if m.bias is not None else torch.zeros(w.shape[0], dtype=torch.float64)
        nxt = mods[i + 1] if i + 1 < len(mods) else None
        if isinstance(nxt, torch.nn.BatchNorm2d):
            scale = nxt.weight.detach().double() / torch.sqrt(nxt.running_var.double() + nxt.eps)
            w = w * scale[:, None, None, None]
            b = (b - nxt.running_mean.double()) * scale + nxt.bias.detach().double()
        layers.append((w, b))
    return layers


def forward_folded(layers, y):
    # plain clamp pad + conv + bias + ReLU stack: what the CUDA kernels compute
    v = y
    for i, (w, b) in enumerate(layers):
        v = F.conv2d(F.pad(v, (1, 1, 1, 1), mode="replicate"), w, b)
        if i < len(layers) - 1:
            v = F.relu(v)
    return y - v


def main(run="depth13"):
    cfg = json.loads((ROOT / "runs" / run / "config.json").read_text())
    model = Denoiser(cfg["depth"], cfg["features"]).eval()
    model.load_state_dict(torch.load(ROOT / "runs" / run / "model.pt", map_location="cpu"))
    out = ROOT / "runs" / run / "export"
    out.mkdir(exist_ok=True)

    layers = fold(model)
    for i, (w, b) in enumerate(layers):
        np.save(out / f"conv{i:02d}_w.npy", w.permute(2, 3, 1, 0).float().contiguous().numpy())  # KCRS -> RSCK
        np.save(out / f"conv{i:02d}_b.npy", b.float().numpy())

    # test input: centre crop of the first validation image, fixed noise
    x = np.load(ROOT / "data" / "gray" / "valid" / "0801.npy")
    h, w = x.shape
    x = x[(h - CROP) // 2:(h + CROP) // 2, (w - CROP) // 2:(w + CROP) // 2]
    y = (x + SIGMA * np.random.default_rng(0).standard_normal(x.shape)).astype(np.float32)
    yt = torch.from_numpy(y)[None, None]
    with torch.no_grad():
        ref = model(yt)[0, 0].numpy()
        folded = forward_folded(layers, yt.double())[0, 0].numpy()
    np.save(out / "input.npy", y)
    np.save(out / "reference.npy", ref)
    print(f"{len(layers)} layers, max abs(folded - model) {np.abs(folded - ref).max():.2e}")


if __name__ == "__main__":
    main()