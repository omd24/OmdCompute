import numpy as np

def coords(W, H):
    # two H×W arrays: row index, column index
    # v, u = np.mgrid[0:2, 0:3]
    #  v = [[0, 0, 0],      u = [[0, 1, 2],
    #       [1, 1, 1]]          [0, 1, 2]]
    v, u = np.mgrid[0:H, 0:W]       
    return u, v     # return two 2D arrays of indices: u (column index), v (row index)

def gradient(W, H):
    u, v = coords(W, H)
    return u / (W - 1)

def checkerboard(W, H, cell):
    u, v = coords(W, H)
    # Note: the .astype(np.float64) is needed bc otherwise // would give an array (unlike / operator which stays as single value)
    return ((u // cell + v // cell) % 2).astype(np.float64)   # floor division via // operator

def zone_plate(W, H, max_frequency):
    u, v = coords(W, H)
    cu, cv = (W - 1) / 2, (H - 1) / 2            # image centre
    r2 = (u - cu) ** 2 + (v - cv) ** 2           # squared distance from the centre
    r_max = np.sqrt(cu ** 2 + cv ** 2)           # distance to a corner
    alpha = np.pi * max_frequency / r_max
    return 0.5 + 0.5 * np.cos(alpha * r2)

def diagonal_gradient(W, H):
    u, v = coords(W, H)
    return 0.5 * (u / (W - 1) + v / (H - 1))

def step_edge(W, H):
    u, v = coords(W, H)
    return (u >= W / 2).astype(np.float64)       # comparison gives a bool array (Iverson bracket)

def flat(W, H, value):
    return np.full((H, W), value, dtype=np.float64)   # note the (H, W) row major

PATTERNS = {
    "gradient": gradient,
    "diagonal_gradient": diagonal_gradient,
    "step_edge": step_edge,
    "checkerboard": checkerboard,
    "flat": flat,
    "zone_plate": zone_plate,
}