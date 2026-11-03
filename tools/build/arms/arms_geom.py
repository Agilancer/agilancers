"""Lance and shield geometry (numpy) for all 10 sheet designs. Blades and
shield faces are extruded from the silhouettes cut out of the sheets, so
each variant's shape matches its texture."""
import os, sys, json
import numpy as np
from scipy import ndimage
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
sys.path.insert(0, os.path.join(HERE, '..', 'hero'))
from weapon_layout import LANCE_RECTS, VARIANTS
from sdf import Grid, marching_cubes, smin, smax, capsule, torus, length

ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
PIECES = os.path.join(ROOT, 'build', 'pieces')
META = json.load(open(os.path.join(PIECES, 'pieces.json')))

# ---------------------------------------------------------------- placement in the hero's bind pose
GRIP = np.array([-0.845, 1.408, -0.068])          # centre of the right fist
AXIS = np.array([-0.3, 0.0, 1.0]); AXIS /= np.linalg.norm(AXIS)   # butt -> tip (thumb side)
LANCE_LEN = 1.75            # one-handed: shorter shaft
GRIP_FROM_BUTT = 0.30       # little shaft behind the fist (the sword animations swing the butt past the head/legs)
GRIP_HALF = 0.17            # ~34 cm wrapped grip
SHIELD_CENTER = np.array([0.600, 1.497, -0.068])  # just above the left forearm (dorsal side)
SHIELD_N = np.array([0.0, 1.0, 0.0])              # face normal
SHIELD_UP = np.array([0.0, 0.0, 1.0])             # top of the design
SHIELD_RIGHT = np.cross(SHIELD_UP, SHIELD_N)      # image right as seen from the front
SHIELD_H = 0.64


def lance_frame():
    """Orthonormal frame (right, axis, normal) for the lance; blade plane = right/axis."""
    up = AXIS
    side = np.cross(up, np.array([0.0, 1.0, 0.0])); side /= np.linalg.norm(side)
    nrm = np.cross(side, up)
    return side, up, nrm


def to_world_lance(L):
    """lance-local (x across blade, y along axis from butt, z blade normal) -> bind space"""
    s, u, n = lance_frame()
    base = GRIP - AXIS * GRIP_FROM_BUTT
    return base + L[:, 0:1] * s + L[:, 1:2] * u + L[:, 2:3] * n


def to_world_shield(L):
    return SHIELD_CENTER + L[:, 0:1] * SHIELD_RIGHT + L[:, 1:2] * SHIELD_UP + L[:, 2:3] * SHIELD_N


# ---------------------------------------------------------------- lathe helper
def lathe(profile, segs=20, v_range=(0, 1)):
    """profile: list of (y, r). Returns V (local), F, and (theta, t) params per vertex."""
    prof = np.array(profile, float)
    n = len(prof)
    th = np.linspace(0, 2 * np.pi, segs + 1)
    V, P = [], []
    for i, (y, r) in enumerate(prof):
        for t in th:
            V.append((np.cos(t) * r, y, np.sin(t) * r))
            P.append((t / (2 * np.pi), i / (n - 1)))
    F = []
    w = segs + 1
    for i in range(n - 1):
        for j in range(segs):
            a, b, c, d = i * w + j, i * w + j + 1, (i + 1) * w + j + 1, (i + 1) * w + j
            F += [(a, d, b), (b, d, c)]
    return np.array(V), np.array(F), np.array(P)


# ---------------------------------------------------------------- 2D silhouette sdf
def mask_sdf(mask, px_per_m):
    m = mask.astype(bool)
    din = ndimage.distance_transform_edt(m)
    dout = ndimage.distance_transform_edt(~m)
    return (dout - din) / px_per_m


def sample2d(img, px, py, cval=1.0):
    return ndimage.map_coordinates(img, [py, px], order=1, mode='constant', cval=cval)


# ---------------------------------------------------------------- blade
def blade_mesh(key):
    m = np.asarray(Image.open(os.path.join(PIECES, f'lance_{key}_blademask.png')).convert('L')) > 127
    x0, y0, x1, y1 = LANCE_RECTS['headA']
    sub = m[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub)
    top, bot = ys.min(), ys.max()
    # drop the socket ring drawn at the bottom of the head (modelled as a lathe instead)
    cut = bot - int(0.10 * (bot - top))
    sub = sub.copy(); sub[cut:] = False
    blade_len = 0.33
    ppm = (cut - top) / blade_len
    sd = mask_sdf(np.pad(sub, 8), ppm)
    cx = (xs.min() + xs.max()) / 2 + 8
    H = sd.shape[0]

    def f(p):
        # local: x across, y up (0 at cut line), z thickness
        px = cx + p[:, 0] * ppm
        py = (cut + 8) - p[:, 1] * ppm
        d2 = sample2d(sd, px, py, cval=0.05)
        inner = np.maximum(-d2, 0)
        half = 0.0012 + np.minimum(inner * 0.32, 0.0055)
        ridge = 0.0018 * np.exp(-(p[:, 0] / 0.004) ** 2)
        d3 = np.abs(p[:, 2]) - (half + ridge)
        return smax(d2, d3, 0.0008)
    wmax = (xs.max() - xs.min()) / 2 / ppm + 0.01
    g = Grid((-wmax, -0.004, -0.012), (wmax, blade_len + 0.01, 0.012), 0.0007)
    V, F = marching_cubes(g, g.evaluate(f))
    # uv: front (z>0) uses head A, back uses head B mirrored
    return V, F, dict(ppm=ppm, cx=cx - 8 + x0, cut=cut + y0, top=top + y0)


def blade_uv(V, F, info, panel_w, panel_h):
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    front = n[:, 2] >= 0
    UV = np.zeros((len(F), 3, 2))
    xA0, _, xA1, _ = LANCE_RECTS['headA']; xB0, _, xB1, _ = LANCE_RECTS['headB']
    for k in range(3):
        p = tri[:, k]
        px = info['cx'] + p[:, 0] * info['ppm']
        py = info['cut'] - p[:, 1] * info['ppm']
        pxb = (info['cx'] - xA0) + xB0 - p[:, 0] * info['ppm'] + ((xB1 - xB0) - (xA1 - xA0)) / 2
        u = np.where(front, px, pxb) / panel_w
        UV[:, k, 0] = u
        UV[:, k, 1] = 1 - py / panel_h
    return UV


# ---------------------------------------------------------------- lance assembly
def rect_uv(rect, P, panel, u_mirror=True, v_scale=1.0):
    """Map lathe params (theta01, t01) into a panel rect (mirror-wrapped around)."""
    x0, y0, x1, y1 = rect
    u = P[:, 0]
    if u_mirror:
        u = 1 - np.abs(2 * u - 1)
    v = P[:, 1] * v_scale
    v = np.abs(((v + 1) % 2) - 1)       # mirror-repeat along the length
    pw, ph = panel
    return np.c_[(x0 + 1 + u * (x1 - x0 - 2)) / pw, 1 - (y1 - 1 - v * (y1 - y0 - 2)) / ph]


def lance_parts():
    """Lathe sections (y from butt). Returns list of (name, profile, rect, v_scale)."""
    R = 0.0158
    return [
        ('butt',   [(0.0, 0.0), (0.004, 0.010), (0.020, 0.0175), (0.050, 0.0195), (0.060, 0.0185)], 'butt', 1.0),
        ('band_lo',[(0.060, 0.0185), (0.064, 0.0192), (0.082, 0.0192), (0.086, R)], 'band2', 1.0),
        ('shaft_lo', [(0.086, R), (max(GRIP_FROM_BUTT - GRIP_HALF - 0.024, 0.090), R)], 'shaft', 0.6),
        ('ring_lo',[(GRIP_FROM_BUTT - GRIP_HALF - 0.024, R), (GRIP_FROM_BUTT - GRIP_HALF - 0.021, 0.0185), (GRIP_FROM_BUTT - GRIP_HALF - 0.003, 0.0185), (GRIP_FROM_BUTT - GRIP_HALF, 0.0172)], 'ring', 1.0),
        ('grip',   [(GRIP_FROM_BUTT - GRIP_HALF, 0.0172), (GRIP_FROM_BUTT + GRIP_HALF, 0.0172)], 'grip', 2.0),
        ('ring_hi',[(GRIP_FROM_BUTT + GRIP_HALF, 0.0172), (GRIP_FROM_BUTT + GRIP_HALF + 0.003, 0.0185), (GRIP_FROM_BUTT + GRIP_HALF + 0.021, 0.0185), (GRIP_FROM_BUTT + GRIP_HALF + 0.024, R)], 'ring2', 1.0),
        ('shaft_hi', [(GRIP_FROM_BUTT + GRIP_HALF + 0.024, R), (LANCE_LEN - 0.43, R * 0.97)], 'shaft', 1.6),
        ('strip',  [(LANCE_LEN - 0.43, R * 0.97), (LANCE_LEN - 0.43 + 0.003, 0.0172), (LANCE_LEN - 0.40, 0.0172), (LANCE_LEN - 0.397, R * 0.97)], 'band1', 1.0),
        ('collar', [(LANCE_LEN - 0.397, R * 0.97), (LANCE_LEN - 0.39, 0.0185), (LANCE_LEN - 0.36, 0.0205), (LANCE_LEN - 0.345, 0.0235), (LANCE_LEN - 0.338, 0.019), (LANCE_LEN - 0.33, 0.012)], 'collar', 1.0),
    ]


def lance_mesh(key, panel=(251, 627)):
    Vs, Fs, UVs = [], [], []
    off = 0
    for name, prof, rect, vs in lance_parts():
        V, F, P = lathe(prof, segs=24)
        uv = rect_uv(LANCE_RECTS[rect], P, panel, v_scale=vs)
        Vs.append(V); Fs.append(F + off); UVs.append(uv[F]); off += len(V)
    # blade sits on top of the collar
    bV, bF, info = blade_mesh(key)
    bUV = blade_uv(bV, bF, info, *panel)
    bV = bV.copy(); bV[:, 1] += LANCE_LEN - 0.335
    Vs.append(bV[:, [0, 1, 2]]); Fs.append(bF + off); UVs.append(bUV)
    V = np.concatenate(Vs); F = np.concatenate(Fs); UV = np.concatenate(UVs)
    return to_world_lance(V), F, UV


# ---------------------------------------------------------------- shield
def shield_mesh(key, panel=(355, 443)):
    meta = META['shield'][key]
    main = np.asarray(Image.open(os.path.join(PIECES, f'shield_{key}_mask_main.png')).convert('L')) > 127
    back = np.asarray(Image.open(os.path.join(PIECES, f'shield_{key}_mask_back.png')).convert('L')) > 127
    mx0, my0, mx1, my1 = meta['bbox']['main']
    bx0, by0, bx1, by1 = meta['bbox']['back']
    ppm = (my1 - my0) / SHIELD_H
    pad = 10
    sd = mask_sdf(np.pad(main, pad), ppm)
    cx, cy = (mx0 + mx1) / 2 + pad, (my0 + my1) / 2 + pad
    W = (mx1 - mx0) / ppm

    def f(p):
        px = cx + p[:, 0] * ppm
        py = cy - p[:, 1] * ppm
        d2 = sample2d(sd, px, py, cval=0.1)
        r2 = (p[:, 0] / (W / 2)) ** 2 + (p[:, 1] / (SHIELD_H / 2)) ** 2
        bulge = 0.030 * (1 - np.clip(r2, 0, 1.5))           # convex front
        inner = np.maximum(-d2, 0)
        rim = 0.0035 * np.exp(-((inner - 0.012) / 0.008) ** 2)
        zc = bulge - 0.006
        half = 0.0065 + rim
        d3 = np.abs(p[:, 2] - zc) - half
        d = smax(d2, d3, 0.0015)
        # forearm straps (rings around the forearm, below the shield's back)
        for sx in (-0.085, 0.075):
            c = np.array([sx * 1.0, 0.0, -0.057])
            q = p - c
            ring = np.sqrt((np.sqrt(q[:, 1] ** 2 + q[:, 2] ** 2) - 0.047) ** 2 + 0) - 0.0035
            ring = smax(ring, np.abs(q[:, 0]) - 0.012, 0.002)
            d = smin(d, ring, 0.004)
        return d
    g = Grid((-W / 2 - 0.02, -SHIELD_H / 2 - 0.02, -0.115), (W / 2 + 0.02, SHIELD_H / 2 + 0.02, 0.05), 0.0018)
    V, F = marching_cubes(g, g.evaluate(f))
    # per-face mapping: front -> main design, back -> back design, rim/straps -> rim strip / grip
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    cen = tri.mean(1)
    strap = cen[:, 2] < -0.018
    front = (n[:, 2] > 0.35) & ~strap
    backf = (n[:, 2] < -0.35) & ~strap
    pw, ph = panel
    UV = np.zeros((len(F), 3, 2))
    rx0, ry0, rx1, ry1 = meta['bbox']['rim']
    gx0, gy0, gx1, gy1 = meta['bbox']['grip']
    bs = (bx1 - bx0) / (mx1 - mx0)
    for k in range(3):
        p = tri[:, k]
        fx = mx0 + (mx1 - mx0) / 2 + p[:, 0] * ppm
        fy = my0 + (my1 - my0) / 2 - p[:, 1] * ppm
        # back view is mirrored left-right
        bxp = (bx0 + bx1) / 2 - p[:, 0] * ppm * bs
        byp = (by0 + by1) / 2 - p[:, 1] * ppm * (by1 - by0) / (my1 - my0)
        ang = (np.arctan2(p[:, 1], p[:, 0]) / (2 * np.pi) * 6) % 1.0
        rxp = rx0 + 2 + (np.clip((p[:, 2] + 0.01) / 0.05, 0, 1)) * (rx1 - rx0 - 4)
        ryp = ry0 + 2 + ang * (ry1 - ry0 - 4)
        sa = (np.arctan2(p[:, 1], p[:, 2] + 0.057) / (2 * np.pi)) % 1.0
        gxp = gx0 + 2 + np.clip((p[:, 0] + 0.1) / 0.2, 0, 1) * (gx1 - gx0 - 4)
        gyp = gy0 + 2 + sa * (gy1 - gy0 - 4)
        u = np.where(front, fx, np.where(backf, bxp, np.where(strap, gxp, rxp)))
        v = np.where(front, fy, np.where(backf, byp, np.where(strap, gyp, ryp)))
        UV[:, k, 0] = u / pw
        UV[:, k, 1] = 1 - v / ph
    return to_world_shield(V), F, UV
