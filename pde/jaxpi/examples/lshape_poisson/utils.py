from functools import lru_cache

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import jax.numpy as jnp


def inside_lshape_np(x, y, tol=1e-12):
    """L-shaped domain [-1,1]^2 \ [0,1]^2 with boundary included."""
    in_square = (x >= -1.0 - tol) & (x <= 1.0 + tol) & (y >= -1.0 - tol) & (y <= 1.0 + tol)
    in_removed_open = (x > 0.0 + tol) & (y > 0.0 + tol)
    return in_square & (~in_removed_open)


def sample_lshape_interior(num_points, seed=0):
    rng = np.random.default_rng(seed)
    pts = []
    while sum(p.shape[0] for p in pts) < num_points:
        cand = rng.uniform(-1.0, 1.0, size=(max(num_points, 4096), 2))
        mask = (cand[:, 0] < 0.0) | (cand[:, 1] < 0.0)
        pts.append(cand[mask])
    return jnp.asarray(np.concatenate(pts, axis=0)[:num_points])


def sample_lshape_boundary(num_points):
    # DeepXDE polygon: (0,0)->(1,0)->(1,-1)->(-1,-1)->(-1,1)->(0,1)->(0,0)
    vertices = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, -1.0], [-1.0, -1.0], [-1.0, 1.0], [0.0, 1.0]],
        dtype=np.float64,
    )
    edges = list(zip(vertices, np.roll(vertices, -1, axis=0)))
    lengths = np.array([np.linalg.norm(b - a) for a, b in edges])
    counts = np.maximum(2, np.round(num_points * lengths / lengths.sum()).astype(int))
    while counts.sum() > num_points:
        counts[np.argmax(counts)] -= 1
    while counts.sum() < num_points:
        counts[np.argmax(lengths)] += 1

    pts = []
    for (a, b), n in zip(edges, counts):
        t = np.linspace(0.0, 1.0, n, endpoint=False)[:, None]
        pts.append(a[None, :] * (1.0 - t) + b[None, :] * t)
    return jnp.asarray(np.concatenate(pts, axis=0)[:num_points])


@lru_cache(maxsize=4)
def finite_difference_reference(n=161):
    """Reference for -Delta u = 1, u=0 on the L-shaped boundary."""
    xs = np.linspace(-1.0, 1.0, n)
    ys = np.linspace(-1.0, 1.0, n)
    h = xs[1] - xs[0]

    unknown = {}
    coords = []
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if not inside_lshape_np(x, y):
                continue
            on_outer = i == 0 or i == n - 1 or j == 0 or j == n - 1
            on_inner = (abs(x) < 1e-12 and y >= 0.0) or (abs(y) < 1e-12 and x >= 0.0)
            if on_outer or on_inner:
                continue
            unknown[(i, j)] = len(coords)
            coords.append((x, y))

    rows, cols, data = [], [], []
    rhs = np.ones(len(coords))
    for (i, j), row in unknown.items():
        rows.append(row)
        cols.append(row)
        data.append(4.0 / h**2)
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            col = unknown.get((ni, nj))
            if col is not None:
                rows.append(row)
                cols.append(col)
                data.append(-1.0 / h**2)

    mat = sp.csr_matrix((data, (rows, cols)), shape=(len(coords), len(coords)))
    sol = spla.spsolve(mat, rhs)

    grid = np.zeros((n, n), dtype=np.float64)
    for (i, j), idx in unknown.items():
        grid[i, j] = sol[idx]

    xx, yy = np.meshgrid(xs, ys, indexing="ij")
    mask = inside_lshape_np(xx, yy)
    pts = np.stack([xx[mask], yy[mask]], axis=-1)
    vals = grid[mask, None]
    return jnp.asarray(pts), jnp.asarray(vals)
