import numpy as np
from PIL import Image

# encoding linear data to srgb to cancel monitor decode
# Side note: monitor always applies srgb decode curve, bc srgb is display standard and that's standard bc 8-bit values are spent more efficiently that way. 
# So anything sent to the screen (aka our 8-bit PNG) must be sRGB-encoded 
# (for an linear format file like exr this manual conversion wouldn't be needed as image viewers automatically take care of linear_to_srgb for such formats)
def linear_to_srgb(x):
    x = np.clip(x, 0.0, 1.0)                     # display only: limit values beyond [0, 1] due to unbound noise
    return np.where(x <= 0.0031308,
                    12.92 * x,
                    1.055 * x ** (1 / 2.4) - 0.055)

def save_png(path, x):
    img8 = np.round(linear_to_srgb(x) * 255).astype(np.uint8)
    Image.fromarray(img8).save(path)             # H×W uint8 to greyscale PNG

def save_error_png(path, e, scale):
    t = np.clip(e / scale, -1.0, 1.0)            # error relative to scale [-1, 1]
    r = np.where(t > 0, 1.0, 1.0 + t)            # fades out for negative errors
    g = 1.0 - np.abs(t)                          # white at 0, fades with |error|
    b = np.where(t < 0, 1.0, 1.0 - t)            # fades out for positive errors
    rgb = np.stack([r, g, b], axis=-1)           # HxWx3colour image
    Image.fromarray(np.round(rgb * 255).astype(np.uint8)).save(path)