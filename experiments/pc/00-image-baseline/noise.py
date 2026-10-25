import numpy as np

def unit_noise(W, H, seed):
    rng = np.random.default_rng([seed, W, H])    # seeded generator; a list seed mixes all three values (fixed noise z per image size)
    return rng.standard_normal((H, W))           # float64, mean 0, variance 1

def add_noise(x, z, sigma):
    return x + sigma * z                         # we're intentionally not bounding the output to hav zero-mean error (i.e., effective noise) over many random samples