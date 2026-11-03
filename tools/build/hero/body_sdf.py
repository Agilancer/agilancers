"""Voxelise the mannequin (per closed shell, ray parity) and turn it into a
smooth signed distance field: the volume all garments are tailored around."""
import os, sys, time
import numpy as np
from scipy import ndimage, sparse
sys.path.insert(0, os.path.dirname(__file__))
from sdf import Grid

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


def weld(V, T, tol=1e-5):
    key = np.round(V / tol).astype(np.int64)
    _, inv, idx = np.unique(key, axis=0, return_inverse=True, return_index=True)[0:3][::-1][::-1], None, None
    uniq, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return V[first], inv.reshape(-1)[T]


def shells(V, T):
    n = len(V)
    e = np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n))
    nc, lab = sparse.csgraph.connected_components(A, directed=False)
    return [T[lab[T[:, 0]] == c] for c in range(nc)]


def voxelize(V, T, grid):
    nx, ny, nz = grid.shape
    inside = np.zeros(grid.shape, bool)
    jit = np.array([1.37e-5, 2.11e-5])
    for S in shells(V, T):
        if len(S) < 4:
            continue
        C = np.zeros((nx, ny, nz + 1), np.int16)
        P = V[S]  # (m,3,3)
        gx = (P[..., 0] - grid.lo[0]) / grid.h + jit[0]
        gy = (P[..., 1] - grid.lo[1]) / grid.h + jit[1]
        for t in range(len(S)):
            x0, x1 = int(np.ceil(gx[t].min())), int(np.floor(gx[t].max()))
            y0, y1 = int(np.ceil(gy[t].min())), int(np.floor(gy[t].max()))
            if x1 < x0 or y1 < y0:
                continue
            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1), indexing='ij')
            xs = xs.ravel(); ys = ys.ravel()
            (ax, bx, cx), (ay, by, cy) = gx[t], gy[t]
            det = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
            if abs(det) < 1e-12:
                continue
            l1 = ((by - cy) * (xs - cx) + (cx - bx) * (ys - cy)) / det
            l2 = ((cy - ay) * (xs - cx) + (ax - cx) * (ys - cy)) / det
            l3 = 1 - l1 - l2
            ok = (l1 >= 0) & (l2 >= 0) & (l3 >= 0)
            if not ok.any():
                continue
            z = l1[ok] * P[t, 0, 2] + l2[ok] * P[t, 1, 2] + l3[ok] * P[t, 2, 2]
            kz = np.clip(np.ceil((z - grid.lo[2]) / grid.h).astype(int), 0, nz)
            np.add.at(C, (xs[ok], ys[ok], kz), 1)
        par = (np.cumsum(C, axis=2)[..., :nz] % 2).astype(bool)
        inside |= par
    return inside


def build(h=0.004, smooth_sigma=1.5):
    d = np.load(os.path.join(ROOT, 'build/mannequin.npz'))
    V, T = weld(d['V'], d['T'])
    lo = V.min(0) - 0.08
    hi = V.max(0) + 0.08
    grid = Grid(lo, hi, h)
    t0 = time.time()
    inside = voxelize(V, T, grid)
    inside = ndimage.binary_fill_holes(inside)
    print('voxelised', grid.shape, inside.sum(), f'{time.time()-t0:.1f}s')
    dout = ndimage.distance_transform_edt(~inside) * h
    din = ndimage.distance_transform_edt(inside) * h
    sdf = (dout - din).astype(np.float32)
    sdf = ndimage.gaussian_filter(sdf, smooth_sigma)
    np.savez_compressed(os.path.join(ROOT, 'build/body_sdf.npz'), sdf=sdf, lo=grid.lo, h=h, shape=np.array(grid.shape))
    return grid, sdf


if __name__ == '__main__':
    grid, sdf = build()
    # report cross-section extents at a few heights (torso only: |x| < 0.25)
    xs, ys, zs = grid.points()
    for y in [0.1, 0.3, 0.53, 0.75, 0.92, 1.0, 1.05, 1.1, 1.2, 1.3, 1.38, 1.44, 1.5, 1.55]:
        j = int(round((y - grid.lo[1]) / grid.h))
        sl = sdf[:, j, :] < 0
        ix, iz = np.nonzero(sl)
        tor = np.abs(xs[ix]) < 0.25
        if tor.any():
            print(f'y={y:.2f}  x[{xs[ix[tor]].min():+.3f},{xs[ix[tor]].max():+.3f}]  z[{zs[iz[tor]].min():+.3f},{zs[iz[tor]].max():+.3f}]')
