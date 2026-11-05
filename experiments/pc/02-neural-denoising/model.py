"""DnCNN-style residual denoiser: the network predicts the noise, x = y - f(y)."""
## the network, which estimates the noise and subtracts it.

import torch.nn as nn


def conv(channels_in, channels_out, bias):
    # replicate padding is the clamp boundary used in 00 and 01
    return nn.Conv2d(channels_in, channels_out, 3, padding=1,
                     padding_mode="replicate", bias=bias)

## number of layers 13 to match NLM radius (search radius of 10 + patch radius of 3)
class Denoiser(nn.Module):
    def __init__(self, depth=13, features=64):
        super().__init__()
        layers = [conv(1, features, True), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [conv(features, features, False),
                       nn.BatchNorm2d(features),
                       nn.ReLU(inplace=True)]
        layers += [conv(features, 1, False)]
        self.noise = nn.Sequential(*layers)

    def forward(self, y):
        return y - self.noise(y)