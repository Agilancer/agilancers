"""Signed-distance-field toolkit (numpy) used to model the hero's body parts
and garments. All coordinates are glTF space: metres, +Y up, +Z forward
(character faces +Z, his left hand is +X)."""
import numpy as np
from scipy import ndimage

# ---------------------------------------------------------------- primitives
def length(v):
    return np.sqrt((v * v).sum(-1))


def sphere(p, c, r):
    return length(p - np.asarray(c)) - r


def ellipsoid(p, c, r):
    """Cheap, reasonably accurate ellipsoid distance (iq's bound)."""
    q = (p - np.asarray(c)) / np.asarray(r)
    k0 = length(q)
    k1 = length(q / np.asarray(r))
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def capsule(p, a, b, ra, rb=None):
    """Round cone between a (radius ra) and b (radius rb)."""
    rb = ra if rb is None else rb
    a = np.asarray(a, float); b = np.asarray(b, float)
    ba = b - a
    pa = p - a
    h = np.clip((pa @ ba) / (ba @ ba), 0, 1)
    r = ra + (rb - ra) * h
    return length(pa - h[:, None] * ba) - r


def box(p, c, half, round_=0.0, rot=None):
    q = p - np.asarray(c)
    if rot is not None:
        q = q @ np.asarray(rot)
    q = np.abs(q) - (np.asarray(half) - round_)
    return length(np.maximum(q, 0)) + np.minimum(q.max(-1), 0) - round_


def plane(p, n, d):
    """Signed distance to plane n.p = d (positive on the +n side)."""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    return p @ n - d


def torus(p, c, axis, R, r):
    axis = np.asarray(axis, float); axis /= np.linalg.norm(axis)
    q = p - np.asarray(c)
    h = q @ axis
    radial = length(q - h[:, None] * axis)
    return np.sqrt((radial - R) ** 2 + h ** 2) - r


# ---------------------------------------------------------------- operators
def smin(a, b, k):
    if k <= 0:
        return np.minimum(a, b)
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b + (a - b) * h - k * h * (1 - h)


def smax(a, b, k):
    return -smin(-a, -b, k)


def union(*ds, k=0.0):
    out = ds[0]
    for d in ds[1:]:
        out = smin(out, d, k)
    return out


# ---------------------------------------------------------------- grids
class Grid:
    """Axis aligned sampling grid: values[ix, iy, iz] at lo + i*h."""

    def __init__(self, lo, hi, h):
        self.lo = np.asarray(lo, float)
        self.h = float(h)
        self.shape = tuple(int(np.ceil((hi[i] - lo[i]) / h)) + 1 for i in range(3))

    def points(self):
        xs = [self.lo[i] + self.h * np.arange(self.shape[i]) for i in range(3)]
        return xs

    def evaluate(self, fn, chunk=2_000_000):
        xs, ys, zs = self.points()
        out = np.empty(self.shape, np.float32)
        nyz = self.shape[1] * self.shape[2]
        step = max(1, chunk // nyz)
        Y, Z = np.meshgrid(ys, zs, indexing='ij')
        for i0 in range(0, self.shape[0], step):
            xi = xs[i0:i0 + step]
            P = np.empty((len(xi), self.shape[1], self.shape[2], 3))
            P[..., 0] = xi[:, None, None]
            P[..., 1] = Y[None]
            P[..., 2] = Z[None]
            out[i0:i0 + step] = fn(P.reshape(-1, 3)).reshape(len(xi), self.shape[1], self.shape[2])
        return out

    def sample(self, values, p, order=1, cval=1.0):
        idx = ((np.asarray(p) - self.lo) / self.h).T
        return ndimage.map_coordinates(values, idx, order=order, mode='constant', cval=cval)


def marching_cubes(grid, values, level=0.0):
    from skimage.measure import marching_cubes as mc
    v, f, n, _ = mc(values, level=level, spacing=(grid.h,) * 3, allow_degenerate=False)
    v = v + grid.lo
    # skimage's default winding already faces outward (towards increasing distance)
    return v.astype(np.float64), f.astype(np.int64)


def gradient(fn, p, eps=1e-3):
    g = np.zeros_like(p)
    for i in range(3):
        d = np.zeros(3); d[i] = eps
        g[:, i] = (fn(p + d) - fn(p - d)) / (2 * eps)
    n = np.linalg.norm(g, axis=1, keepdims=True)
    return g / np.maximum(n, 1e-9)


def project_to_surface(fn, p, iters=4):
    """Newton-style projection of points onto the zero level set."""
    for _ in range(iters):
        d = fn(p)
        g = gradient(fn, p)
        p = p - d[:, None] * g
    return p
