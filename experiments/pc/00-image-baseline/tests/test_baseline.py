import numpy as np
from fixtures import PATTERNS
from noise import unit_noise, add_noise
from filters import gaussian_kernel_1d, gaussian_direct, gaussian_separable
from metrics import psnr

SIZES = [(256, 256), (257, 129)]
SIGMAS = [0.5, 1.0, 2.0, 4.0]

def test_kernel_normalised_and_symmetric():
    for s in SIGMAS:
        w = gaussian_kernel_1d(s, 3.0)
        assert abs(w.sum() - 1.0) < 1e-12
        assert np.allclose(w, w[::-1])   # symmetry check

def test_separable_equals_direct():
    for W, H in SIZES:
        y = add_noise(PATTERNS["zone_plate"](W, H, max_frequency=0.5), unit_noise(W, H, 1), 0.1)
        for s in SIGMAS:
            d = gaussian_direct(y, s, 3.0)
            sep = gaussian_separable(y, s, 3.0)
            assert np.abs(d - sep).max() < 1e-12

def test_flat_is_unchanged():
    x = PATTERNS["flat"](257, 129, value=0.5)
    assert np.abs(gaussian_separable(x, 2.0, 3.0) - x).max() < 1e-12

def test_ramp_interior_is_preserved():
    for W, H in SIZES:
        x = PATTERNS["diagonal_gradient"](W, H)
        for s in SIGMAS:
            R = len(gaussian_kernel_1d(s, 3.0)) // 2
            e = gaussian_separable(x, s, 3.0) - x
            assert np.abs(e[R:-R, R:-R]).max() < 1e-12     # interior only bc borders bend due to clamping

def test_flat_noise_mse_matches_theory():
    for W, H in SIZES:
        x = PATTERNS["flat"](W, H, value=0.5)
        y = add_noise(x, unit_noise(W, H, 1234), 0.1)
        for s in SIGMAS:
            w = gaussian_kernel_1d(s, 3.0)
            R = len(w) // 2
            e = (gaussian_separable(y, s, 3.0) - x)[R:-R, R:-R]
            expected = 0.1**2 * (w**2).sum() ** 2
            assert abs(np.mean(e**2) / expected - 1) < 0.10

def test_psnr_values():
    x = np.zeros((4, 4))
    assert psnr(x, x) == np.inf
    assert abs(psnr(x + 0.1, x) - 20.0) < 1e-9            # RMS 0.1 -> 20 dB

def test_noise_is_reproducible():
    assert np.array_equal(unit_noise(64, 32, 7), unit_noise(64, 32, 7))
    assert not np.array_equal(unit_noise(64, 32, 7), unit_noise(64, 32, 8))