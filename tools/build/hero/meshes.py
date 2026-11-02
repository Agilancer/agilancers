"""Turn the part SDFs into triangle meshes (marching cubes) and build the
explicit cloth grid for the simulated coat skirt."""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sdf import Grid, marching_cubes, project_to_surface, length
import parts as P


def mc(fn, box, h):
    t = time.time()
    g = Grid(box[0], box[1], h)
    vals = g.evaluate(fn)
    V, F = marching_cubes(g, vals)
    print(f'  mc {g.shape} -> {len(V)} verts, {len(F)} tris ({time.time()-t:.1f}s)')
    return V, F


def mirror_mesh(V, F):
    V2 = V.copy(); V2[:, 0] *= -1
    return V2, F[:, ::-1].copy()


def remove_cap_faces(V, F, plane_axis, value, side, tol=0.0025, nmin=0.9):
    """Delete faces lying on a cut plane (e.g. hems) so garments are open."""
    tri = V[F]
    c = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    on = np.abs(c[:, plane_axis] - value) < tol
    facing = n[:, plane_axis] * side > nmin
    return F[~(on & facing)]


# ------------------------------------------------------------------ free-hanging coat body (cloth grid)
SKIRT_ROWS = 36
SKIRT_HALF_COLS = 28          # per side, front edge .. back centre (inclusive)
VENT_TOP = 0.735              # back vent opens below this height


def _surface_point(t, y, fn, offset):
    """March from the torso axis along the ellipse direction t at height y
    until fn(p) == -offset (cloth mid-surface)."""
    zc, a, b = P.ref_ellipse(np.array([y]))
    d = np.array([np.sin(t) * a[0], 0, np.cos(t) * b[0]]); d /= np.linalg.norm(d)
    o = np.array([0, y, zc[0]])
    rs = np.linspace(0.04, 0.40, 720)
    pts = o + rs[:, None] * d
    val = fn(pts) + offset
    k = int(np.argmax(val > 0))
    return pts[k], d


def skirt_grid():
    """Cloth grid for the whole lower coat: hangs from the yoke seam (front
    shoulders, armpits, shoulder blades) to the knee, open at the front,
    with a back vent. Row 0 is pinned to the skinned yoke at runtime."""
    rows, half = SKIRT_ROWS, SKIRT_HALF_COLS
    cols = half * 2
    th_top = float(P.coat_open_angle(np.array([1.44]))[0])
    f = np.linspace(0, 1, half)
    hem = P.COAT_HEM
    V = np.zeros((rows, cols, 3))
    T = np.zeros((rows, cols))
    # rows are denser near the shoulders where the drape curves most
    srow = np.linspace(0, 1, rows) ** 1.12
    ref_y = 1.06
    for j in range(cols):
        side = -1 if j < half else 1                      # wearer's right half first
        fj = f[j] if j < half else f[::-1][j - half]
        t_top = side * (th_top + fj * (np.pi - th_top))
        y0 = float(P.yoke_line(np.array([t_top]))[0]) + 0.028     # sewn well up under the yoke
        for i in range(rows):
            y = y0 + (hem - y0) * srow[i]
            th = float(P.coat_open_angle(np.array([y]))[0])
            t = side * (th + fj * (np.pi - th))
            T[i, j] = t
            if y > ref_y:
                pt, d = _surface_point(t, y, P.coat_outer_torso, P.COAT_T * 0.5)
            else:
                pt, d = _surface_point(t, ref_y, P.coat_outer_torso, P.COAT_T * 0.5)
                s = (ref_y - y) / (ref_y - hem)
                pt = pt.copy(); pt[1] = y
                pt[0] += d[0] * (0.020 * s + 0.012 * s * s)
                pt[2] += d[2] * ((0.010 if d[2] > 0 else 0.045) * s)
            V[i, j] = pt
    flat = V.reshape(-1, 3)
    for _ in range(30):                               # rest pose clear of hips/thighs
        dd = P.body(flat) - 0.036
        bad = (dd < 0) & (flat[:, 1] < 1.15)          # hips/thighs only (arms are not obstacles here)
        bad[:cols] = False
        if not bad.any():
            break
        dirn = flat[bad].copy(); dirn[:, 1] = 0; dirn[:, 2] -= 0.004
        dirn /= np.maximum(np.linalg.norm(dirn, axis=1, keepdims=True), 1e-9)
        flat[bad] += dirn * (-dd[bad])[:, None] * 0.6
    V = flat.reshape(rows, cols, 3)
    F = []
    for i in range(rows - 1):
        for j in range(cols - 1):
            if j == half - 1:                           # zero-width back-centre seam (vent)
                continue
            a_, b_, c_, d_ = i * cols + j, i * cols + j + 1, (i + 1) * cols + j + 1, (i + 1) * cols + j
            F.append((a_, b_, d_)); F.append((b_, c_, d_))
    F = np.array(F)
    ymid = V[:, half - 1, 1]
    vent_row = int(np.argmax(ymid < VENT_TOP))
    meta = {'rows': rows, 'cols': cols, 'ventCol': half - 1, 'ventRow': vent_row}
    return V.reshape(-1, 3), F, meta
