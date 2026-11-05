from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from prepare import DATA, load_gray, srgb_to_linear
from dataset import PatchDataset, add_noise
from model import Denoiser

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
needs_data = pytest.mark.skipif(not (DATA / "gray").exists(), reason="run prepare.py first")


def test_srgb_to_linear_matches_standard():
    for s, l in [(0.0, 0.0), (0.04045, 0.0031308), (0.5, 0.2140411), (1.0, 1.0)]:
        assert abs(float(srgb_to_linear(np.float32(s))) - l) < 1e-5


def test_luma_of_uniform_colours(tmp_path):
    for colour, expected in [((255, 255, 255), 1.0), ((255, 0, 0), 0.2126), ((0, 255, 0), 0.7152)]:
        Image.new("RGB", (8, 8), colour).save(tmp_path / "c.png")
        assert np.allclose(load_gray(tmp_path / "c.png"), expected, atol=1e-5)


@needs_data
def test_prepared_files_match_png():
    for split, name in [("train", "0001"), ("train", "0800"), ("valid", "0801"), ("valid", "0900")]:
        stored = np.load(DATA / "gray" / split / f"{name}.npy")
        assert np.array_equal(stored, load_gray(DATA / f"DIV2K_{split}_HR" / f"{name}.png"))


@needs_data
def test_split_is_by_image():
    train = {f.stem for f in (DATA / "gray" / "train").glob("*.npy")}
    valid = {f.stem for f in (DATA / "gray" / "valid").glob("*.npy")}
    assert len(train) == 800 and len(valid) == 100 and not (train & valid)


def test_patch_is_a_window_of_its_image(tmp_path):
    # every pixel value is unique, so a patch locates itself in the image
    image = np.arange(100 * 130, dtype=np.float32).reshape(100, 130)
    np.save(tmp_path / "a.npy", image)
    ds = PatchDataset(tmp_path, patch=16, patches_per_image=4)
    assert len(ds) == 4
    for i in range(20):
        p = ds[i % 4]
        assert p.shape == (1, 16, 16) and p.dtype == torch.float32
        y, x = divmod(int(p[0, 0, 0]), 130)
        assert torch.equal(p[0], torch.from_numpy(image[y:y + 16, x:x + 16]))


def test_noise_statistics():
    x = torch.full((64, 1, 64, 64), 0.5, device=DEVICE)
    z = add_noise(x, 0.1) - x
    assert abs(z.mean().item()) < 1e-3 and abs(z.std().item() - 0.1) < 1e-3
    assert not torch.equal(add_noise(x, 0.1), add_noise(x, 0.1))


def test_parameter_count():
    params = sum(p.numel() for p in Denoiser().parameters())
    assert params == (9 * 64 + 64) + 11 * (9 * 64 ** 2 + 2 * 64) + 9 * 64


def test_output_size_follows_input():
    model = Denoiser().to(DEVICE)
    for h, w in [(64, 64), (128, 300)]:
        assert model(torch.zeros(2, 1, h, w, device=DEVICE)).shape == (2, 1, h, w)


def test_receptive_field():
    for depth in (5, 13):
        model = Denoiser(depth).to(DEVICE).eval()
        y = torch.rand(1, 1, 61, 61, device=DEVICE, requires_grad=True)
        model(y)[0, 0, 30, 30].backward()
        touched = (y.grad[0, 0] != 0).nonzero()
        extent = touched.max(dim=0).values - touched.min(dim=0).values + 1
        assert extent.tolist() == [2 * depth + 1] * 2


def test_zero_noise_estimate_returns_input():
    # with the last layer zeroed f(y) = 0, so only the skip connection remains
    model = Denoiser().to(DEVICE).eval()
    torch.nn.init.zeros_(model.noise[-1].weight)
    y = torch.rand(1, 1, 32, 32, device=DEVICE)
    assert torch.equal(model(y), y)