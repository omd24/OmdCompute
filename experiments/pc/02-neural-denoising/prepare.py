"""Converts the DIV2K PNG images to linear one-channel float32 .npy files."""
from pathlib import Path

import numpy as np
from PIL import Image

DATA = Path(__file__).parent / "data"
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)


def srgb_to_linear(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def load_gray(path):
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return (srgb_to_linear(rgb) @ LUMA).astype(np.float32)


def main():
    for split in ("train", "valid"):
        out = DATA / "gray" / split
        out.mkdir(parents=True, exist_ok=True)
        pngs = sorted((DATA / f"DIV2K_{split}_HR").glob("*.png"))
        for png in pngs:
            np.save(out / f"{png.stem}.npy", load_gray(png))
        print(split, len(pngs))


if __name__ == "__main__":
    main()