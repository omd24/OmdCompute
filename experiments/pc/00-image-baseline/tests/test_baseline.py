import numpy as np
from fixtures import PATTERNS
from noise import unit_noise, add_noise
from filters import blur_h, blur_v, gaussian_kernel_1d, gaussian_direct, gaussian_separable, nlm
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

def nlm_brute_force(y, sigma_n, S, P, h_factor):     # per-pixel definition, shares no code with nlm()
    H, W = y.shape
    h = h_factor * sigma_n
    def c(v, u):
        return y[min(max(v, 0), H - 1), min(max(u, 0), W - 1)]
    out = np.zeros_like(y)
    for v in range(H):
        for u in range(W):
            acc, total = 0.0, 0.0
            for dj in range(-S, S + 1):
                for di in range(-S, S + 1):
                    d2 = 0.0
                    for kj in range(-P, P + 1):
                        for ki in range(-P, P + 1):
                            d2 += (c(v + kj, u + ki) - c(v + dj + kj, u + di + ki)) ** 2
                    d2 /= (2 * P + 1) ** 2
                    w = np.exp(-max(d2 - 2 * sigma_n**2, 0.0) / h**2)
                    acc += w * c(v + dj, u + di)
                    total += w
            out[v, u] = acc / total
    return out

def test_nlm_matches_brute_force():
    W, H = 17, 11                                     # small and non-square, so borders and axes matter
    y = add_noise(PATTERNS["zone_plate"](W, H, max_frequency=0.5), unit_noise(W, H, 1), 0.1)
    assert np.abs(nlm(y, 0.1, 2, 1, 1.4) - nlm_brute_force(y, 0.1, 2, 1, 1.4)).max() < 1e-12

def test_nlm_flat_is_unchanged():
    x = PATTERNS["flat"](257, 129, value=0.5)
    assert np.abs(nlm(x, 0.1, 3, 1, 1.4) - x).max() < 1e-12

def test_nlm_large_h_is_a_box_filter():
    W, H = 64, 48                                     # all weights -> 1
    y = add_noise(PATTERNS["zone_plate"](W, H, max_frequency=0.5), unit_noise(W, H, 1), 0.1)
    S = 3
    box = np.full(2 * S + 1, 1.0 / (2 * S + 1))
    assert np.abs(nlm(y, 0.1, S, 1, 1e6) - blur_v(blur_h(y, box), box)).max() < 1e-9

def test_nlm_transpose_symmetry():
    W, H = 64, 48
    y = add_noise(PATTERNS["zone_plate"](W, H, max_frequency=0.5), unit_noise(W, H, 1), 0.1)
    assert np.abs(nlm(y.T, 0.1, 3, 1, 1.4).T - nlm(y, 0.1, 3, 1, 1.4)).max() < 1e-12

def test_nlm_beats_best_gaussian():
    for W, H in SIZES:
        z = unit_noise(W, H, 1234)
        for x in [PATTERNS["step_edge"](W, H), PATTERNS["zone_plate"](W, H, max_frequency=0.5)]:
            y = add_noise(x, z, 0.1)
            best = max(psnr(gaussian_separable(y, s, 3.0), x) for s in SIGMAS)
            assert psnr(nlm(y, 0.1, 10, 3, 1.4), x) > best + 3.0