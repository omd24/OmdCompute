"""Random clean patches from the prepared images. Noise is added on the GPU."""
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class PatchDataset(Dataset):
    def __init__(self, folder, patch=64, patches_per_image=16):
        self.images = [np.load(f) for f in sorted(Path(folder).glob("*.npy"))]
        self.patch = patch
        self.patches_per_image = patches_per_image

    def __len__(self):
        return len(self.images) * self.patches_per_image

    def __getitem__(self, index):
        image = self.images[index % len(self.images)]
        h, w = image.shape
        y = np.random.randint(h - self.patch + 1)
        x = np.random.randint(w - self.patch + 1)
        crop = image[y:y + self.patch, x:x + self.patch].copy()
        return torch.from_numpy(crop)[None]  # 1 x H x W


def add_noise(x, sigma):
    return x + sigma * torch.randn_like(x)