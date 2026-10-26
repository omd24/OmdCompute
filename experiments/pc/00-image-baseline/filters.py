import numpy as np

def gaussian_kernel_1d(sigma, truncate):
    R = int(np.ceil(truncate * sigma))          # blue radius (aka standard deviation aka sigma) beyond 3*sigma is negligible so truncating
    i = np.arange(-R, R + 1)                    # [-R, ..., R]
    w = np.exp(-i**2 / (2 * sigma**2))
    return w / w.sum()                          # normalised: sum = 1

def no_filter(y):
    return y.copy()

def gaussian_direct(y, sigma, truncate):
    w1 = gaussian_kernel_1d(sigma, truncate)
    w2 = np.outer(w1, w1)                        # 2D kernel: w2[j, i] = w1[j] * w1[i]
    R = len(w1) // 2
    p = np.pad(y, R, mode="edge")                # add borders of R pixels around the image (edge = clamp)
    H, W = y.shape
    out = np.zeros_like(y)
    for j in range(-R, R + 1):                   # vertical tap offset
        for i in range(-R, R + 1):               # horizontal tap offset
            # NOTE: unlike a pixel shader, here we move the image around for each kernel, instead of moving the kernel around for each pixel
            out += w2[j + R, i + R] * p[R + j : R + j + H, R + i : R + i + W]
    return out

def blur_h(y, w):
    R = len(w) // 2
    H, W = y.shape
    p = np.pad(y, ((0, 0), (R, R)), mode="edge")   # pad left/right only
    out = np.zeros_like(y)
    for i in range(-R, R + 1):
        out += w[i + R] * p[:, R + i : R + i + W]  # all rows, columns shifted by i
    return out

def blur_v(y, w):
    R = len(w) // 2
    H, W = y.shape
    p = np.pad(y, ((R, R), (0, 0)), mode="edge")   # pad top/bottom only
    out = np.zeros_like(y)
    for j in range(-R, R + 1):
        out += w[j + R] * p[R + j : R + j + H, :]  # rows shifted by j, all columns
    return out

def gaussian_separable(y, sigma, truncate):
    w1 = gaussian_kernel_1d(sigma, truncate)
    return blur_v(blur_h(y, w1), w1)

FILTERS = {
    "no_filter": no_filter,
    "gaussian_direct": gaussian_direct,
    "gaussian_separable": gaussian_separable,
}

# Takes a filter config and returns all variants of the filter based on different filter settings
# each variant is a tuple of (label, filter_fn, filter_params)
def filter_variants(fcfg):
    variants = []

    # 1. no-filter aka identity filter
    if "no_filter" in fcfg:
        variants.append(("no_filter", FILTERS["no_filter"], {}))

    # 2. gaussian filters
    g = fcfg["gaussian"]
    for impl in g["implementations"]:                                       # "direct", "separable"
        for s in g["sigmas"]:                                               # 0.5, 1.0, 2.0, 4.0
            variants.append((f"gaussian_{impl}_s{s}",                       # formatted label for file names and metrics
                             FILTERS[f"gaussian_{impl}"],                   # filter function call
                             {"sigma": s, "truncate": g["truncate"]}))      # filter params
            
    return variants