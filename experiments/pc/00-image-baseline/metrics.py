import numpy as np

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b, peak=1.0):
    e = mse(a, b)
    return np.inf if e == 0 else 10 * np.log10(peak**2 / e)