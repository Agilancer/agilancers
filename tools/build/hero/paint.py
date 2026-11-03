"""Procedural projection of the provided texture pieces onto the hero's
surfaces. Every function takes surface points P (n,3) and normals N (n,3)
in glTF space and returns linear-ish sRGB colours (n,3) in [0,1]; bake.py
rasterises them into per-part UV atlases."""
import os, sys, json
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import parts as P
from texlib import extend_band

PIECES = os.path.join(P.ROOT, 'build', 'pieces')
P.place_bun(); P._MH['bun'] = True
LM = json.load(open(os.path.join(P.ROOT, 'build', 'cache', 'mh_landmarks.json')))
_mhv = np.load(os.path.join(P.ROOT, 'build', 'cache', 'mh_head.npz'))['V']
META = json.load(open(os.path.join(PIECES, 'pieces.json')))['outfit']
_cache = {}


def piece(name):
    if name not in _cache:
        from PIL import Image
        a = np.asarray(Image.open(os.path.join(PIECES, name + '.png'))).astype(np.float32) / 255.0
        _cache[name] = a
    return _cache[name]


def sample(img, px, py):
    """Bilinear sample (clamped) of an (H,W,C) image at pixel centres px,py."""
    H, W = img.shape[:2]
    x = np.clip(px - 0.5, 0, W - 1.001)
    y = np.clip(py - 0.5, 0, H - 1.001)
    x0 = np.floor(x).astype(int); y0 = np.floor(y).astype(int)
    fx = (x - x0)[:, None]; fy = (y - y0)[:, None]
    c00 = img[y0, x0]; c10 = img[y0, x0 + 1]; c01 = img[y0 + 1, x0]; c11 = img[y0 + 1, x0 + 1]
    return (c00 * (1 - fx) + c10 * fx) * (1 - fy) + (c01 * (1 - fx) + c11 * fx) * fy


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def mix(a, b, t):
    t = np.asarray(t)[:, None]
    return a * (1 - t) + b * t


def flatten_lowfreq(img, sigma=10):
    """Remove large-scale shading so a band can be repeated without banding."""
    rgb = img[..., :3]
    low = np.stack([ndimage.gaussian_filter(rgb[..., c], sigma) for c in range(3)], -1)
    mean = rgb.reshape(-1, 3).mean(0)
    out = img.copy()
    out[..., :3] = np.clip(rgb - low + mean, 0, 1)
    return out


def ellipse_arc(t, a, b, n=64):
    """Arc length along ellipse (x=a sin t, z=b cos t) from t=0 to t (vectorised, signed)."""
    ts = np.linspace(0, 1, n)[None, :] * np.abs(t)[:, None]
    dx = (a[:, None] * np.cos(ts)); dz = (b[:, None] * np.sin(ts))
    sp = np.sqrt(dx * dx + dz * dz)
    s = np.trapezoid(sp, ts, axis=1)
    return np.sign(t) * s


# ====================================================================== HEAD / SKIN
FACE_CX, FACE_CY, FACE_S = 169.0, 138.0, 1667.0
SKIN_TONE = None


def skin_tone():
    global SKIN_TONE
    if SKIN_TONE is None:
        img = piece('head')
        # the photo's own average skin (forehead + both cheeks) so the seam disappears
        f = face_photo()
        c = np.concatenate([f[150:200, 200:300, :3].reshape(-1, 3), f[370:420, 105:160, :3].reshape(-1, 3),
                            f[370:420, 340:395, :3].reshape(-1, 3)]).mean(0)
        SKIN_TONE = c
    return SKIN_TONE


# face photo landmarks (piece pixel rows) <-> MakeHuman landmarks (height)
_ROWS = [114.0, 138.0, 203.0, 249.0, 332.0]


def _photo_row(y):
    ys = np.array([LM['glabella'][1], P.Y_EYE, LM['subnasale'][1], LM['stomion'][1], LM['menton'][1]])
    rows = np.array(_ROWS)
    o = np.argsort(ys)
    r = np.interp(y, ys[o], rows[o])
    # extrapolate with the photo's own scale above the brow / below the chin
    r = np.where(y > ys.max(), rows[0] - (y - ys.max()) * FACE_S, r)
    r = np.where(y < ys.min(), rows[-1] + (ys.min() - y) * FACE_S, r)
    return r


# The face texture comes from the client's own frontal video frame
# (assets/source/head_reference/face_front.png, tone-mapped from HDR, 500x675,
# ~3016 px per metre). Landmarks measured on that frame:
CLIENT = dict(path='assets/source/head_reference/face_front.png', cx=250.0, eye_row=301.5, s=3016.0,
              oval=(250.0, 420.0, 205.0, 265.0), forehead=(95.0, 125.0),
              eyes=((151.5, 301.5), (341.5, 301.5)), eye_half=(44.0, 11.5),
              nostrils=((212.0, 431.0), (287.0, 431.0)), nostril_half=(19.0, 8.0), delight=0.45, sat=0.85)
_TW = json.load(open(os.path.join(os.path.dirname(__file__), 'face_tweaks.json')))
CLIENT.update({k: (tuple(v) if isinstance(v, list) else v) for k, v in _TW.get('texture', {}).items() if k != 'eye_half'})
if 'eye_half' in _TW.get('texture', {}):
    CLIENT['eye_half'] = tuple(_TW['texture']['eye_half'])
CLIENT_LM = {
    'eye_r': (151.5, 301.5), 'eye_l': (341.5, 301.5),
    'canthus_out_r': (108.5, 304), 'canthus_in_r': (193.5, 304), 'canthus_in_l': (301.5, 304), 'canthus_out_l': (391.5, 301.5),
    'glabella': (250, 254), 'tip': (250, 387.5), 'subnasale': (250, 440), 'ala_r': (193.5, 429), 'ala_l': (304, 429),
    'stomion': (250, 500), 'mouth_r': (158.5, 500), 'mouth_l': (341.5, 500), 'menton': (250, 641.5),
    'gonion_r': (54, 508.5), 'gonion_l': (475, 508.5),
    'brow_in_r': (204, 248.5), 'brow_in_l': (291.5, 241.5), 'brow_out_r': (83.5, 254), 'brow_out_l': (416.5, 246),
}

# (stock face photo from the texture sheet, no longer used)
PHOTO_LM = {
    'eye_r': (116.5, 138), 'eye_l': (221.5, 138),
    'canthus_out_r': (91.5, 139), 'canthus_in_r': (144, 140), 'canthus_in_l': (196.5, 140), 'canthus_out_l': (248, 138),
    'glabella': (169, 116), 'tip': (169, 194), 'subnasale': (169, 203), 'ala_r': (145.5, 201.5), 'ala_l': (195, 201.5),
    'stomion': (169, 249), 'mouth_r': (124, 249), 'mouth_l': (214, 249), 'menton': (169, 332),
    'gonion_r': (54, 284), 'gonion_l': (284, 284),
    'brow_in_r': (144, 113), 'brow_in_l': (195, 113), 'brow_out_r': (66, 119), 'brow_out_l': (272, 119),
}


def _unwrap_x(x):
    return x * (1 + 0.16 * np.clip(np.abs(x) / 0.075, 0, 1.3) ** 2)


_TPS = {}


def face_warp(xu, y):
    """Thin-plate spline from (unwrapped x, height) on the MakeHuman head to
    face-photo pixels, pinned on 17 facial landmarks plus a ring of anchors
    that keep the old linear mapping away from the face."""
    if 'f' not in _TPS:
        from scipy.interpolate import RBFInterpolator
        src, dst = [], []
        lmx = dict(LM)
        ey = P.Y_EYE
        for side, sg in (('r', -1), ('l', 1)):
            lmx.setdefault(f'brow_in_{side}', [sg * 0.0160, ey + 0.0165, 0])
            lmx.setdefault(f'brow_out_{side}', [sg * 0.0525, ey + 0.0128, 0])
        for k, (px, py) in CLIENT_LM.items():
            p = lmx[k]
            src.append((_unwrap_x(np.array([p[0]]))[0], p[1])); dst.append((px, py))
        for a in np.linspace(0, 2 * np.pi, 20, endpoint=False):
            xu_, yy = 0.105 * np.cos(a), 1.655 + 0.13 * np.sin(a)
            src.append((xu_, yy)); dst.append((CLIENT['cx'] + CLIENT['s'] * xu_, CLIENT['eye_row'] + CLIENT['s'] * (P.Y_EYE - yy)))
        _TPS['f'] = RBFInterpolator(np.array(src), np.array(dst), kernel='thin_plate_spline', smoothing=1e-7)
    return _TPS['f'](np.c_[xu, y])


def skin_detail(Pt, scale=1.0):
    """Fine skin texture (pores) from a clean forehead patch, tiled with offsets."""
    if 'skin_hp' not in _cache:
        patch = face_photo()[150:215, 190:310, :3].mean(2)
        _cache['skin_hp'] = patch - ndimage.gaussian_filter(patch, 3)
    hp = _cache['skin_hp']
    H, W = hp.shape
    u = (Pt[:, 0] * 2.1 + Pt[:, 2] * 1.3) * 3016 * scale
    v = (Pt[:, 1] + Pt[:, 2] * 0.4) * 3016 * scale
    def mt(c, n):
        k = np.floor(c / n); f = c - k * n
        return np.where(k % 2 == 0, f, n - 1 - f).astype(int)
    return hp[mt(v, H - 1), mt(u, W - 1)]


def face_photo():
    """The client's face frame with its eye openings and nostrils painted over
    with surrounding skin (the real MakeHuman eyeballs / 3D nostrils show
    instead, so nothing is doubled) and a soft lash line along each upper lid."""
    if 'face' in _cache:
        return _cache['face']
    from PIL import Image
    rgb = np.asarray(Image.open(os.path.join(P.ROOT, CLIENT['path'])).convert('RGB')).astype(np.float32) / 255
    H, W = rgb.shape[:2]
    # partially remove the baked overhead lighting: divide out very low-frequency luminance
    lum = rgb.mean(2)
    low = ndimage.gaussian_filter(lum, 28)
    ref = float(np.median(lum[150:560, 90:410]))
    rgb = np.clip(rgb * ((ref / np.maximum(low, 0.05)) ** CLIENT.get('delight', 0.45))[..., None], 0, 1)
    # tame the camera's red push; overall gain
    m = rgb.mean(2, keepdims=True)
    rgb = np.clip(m + (rgb - m) * CLIENT.get('sat', 0.85), 0, 1)
    rgb[..., 0] = np.clip(rgb[..., 0] * CLIENT.get('red', 1.0) + (1 - CLIENT.get('red', 1.0)) * rgb[..., 1], 0, 1)
    rgb = np.clip(rgb * CLIENT.get('gain', 1.0), 0, 1)
    # darker/lighter brows (region above each eye)
    if CLIENT.get('brows', 1.0) != 1.0:
        yy0, xx0 = np.mgrid[0:H, 0:W].astype(np.float32)
        for cx, cy in CLIENT['eyes']:
            e = np.clip(1.15 - (((xx0 - cx) / 62.0) ** 2 + ((yy0 - cy + 52) / 20.0) ** 2), 0, 1)
            dark = np.clip(0.45 - rgb.mean(2), 0, 1) / 0.45            # only the dark brow hairs
            rgb *= (1 - e * dark * (1 - CLIENT['brows']))[..., None] if CLIENT['brows'] > 1 else 1
            if CLIENT['brows'] < 1:
                rgb += (e * dark * (1 - CLIENT['brows']) * 0.3)[..., None]
        rgb = np.clip(rgb, 0, 1)
    img = np.dstack([rgb, np.ones((H, W), np.float32)])
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    hole = np.zeros((H, W), bool)
    lash = np.zeros((H, W), np.float32)
    ax, ay = CLIENT['eye_half']
    for cx, cy in CLIENT['eyes']:
        hole |= ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 < 1.0
        top = np.exp(-((yy - (cy - ay * np.sqrt(np.clip(1 - ((xx - cx) / ax) ** 2, 0, 1)))) / 2.2) ** 2) * (np.abs(xx - cx) < ax - 1)
        lash = np.maximum(lash, top)
    nx, ny = CLIENT['nostril_half']
    for cx, cy in CLIENT['nostrils']:
        hole |= ((xx - cx) / nx) ** 2 + ((yy - cy) / ny) ** 2 < 1.0
    # under-nose strip (selfie from below shows the nostril shadows there) and the alar crease slits
    hole |= ((xx - 250) / 48.0) ** 2 + ((yy - 440) / 9.0) ** 2 < 1.0
    for cx in (176.0, 324.0):
        hole |= ((xx - cx) / 9.0) ** 2 + ((yy - 424) / 16.0) ** 2 < 1.0
    # red blemishes: small spots much redder than their surroundings
    redness = rgb[..., 0] - 0.5 * (rgb[..., 1] + rgb[..., 2])
    spot = (redness - ndimage.gaussian_filter(redness, 10)) > 0.07
    spot = ndimage.binary_opening(spot)
    lab, nl = ndimage.label(spot)
    if nl:
        sizes = ndimage.sum(spot, lab, range(1, nl + 1))
        spot = np.isin(lab, 1 + np.nonzero(sizes < 90)[0])           # only small blemishes
    keep_out = (((xx - 250) / 105.0) ** 2 + ((yy - 500) / 32.0) ** 2 < 1.0) | (((xx - 250) / 48.0) ** 2 + ((yy - 392) / 42.0) ** 2 < 1.0)
    hole |= ndimage.binary_dilation(spot & ~keep_out, iterations=3)
    rgba = img.copy(); rgba[..., 3] = (~hole).astype(np.float32)
    from texlib import dilate_colors
    filled = dilate_colors(rgba, iters=60)
    filled = np.stack([ndimage.gaussian_filter(filled[..., c], 2.0) for c in range(3)], -1)
    out = img.copy()
    out[..., :3] = np.where(hole[..., None], filled, img[..., :3])
    out[..., :3] *= (1 - 0.5 * lash)[..., None]
    # soften dark lower-lid shadows (geometry + AO provide them)
    for cx, cy in CLIENT['eyes']:
        e = ((xx - cx) / (ax + 6)) ** 2 + ((yy - cy - 22) / 14.0) ** 2
        w = np.clip(1.2 - e, 0, 1)[..., None] * 0.45
        local = ndimage.gaussian_filter(out[..., :3], (12, 12, 0))[int(cy + 40), int(cx)]
        out[..., :3] = out[..., :3] * (1 - w) + local * w
    _cache['face'] = out
    return out


def buzz_pattern(Pt):
    """High-pass of the client's side-of-head video frame (short hair), tiled
    around the head with mirrored offsets; returns signed detail ~[-0.2, 0.2]."""
    if 'buzz' not in _cache:
        from PIL import Image
        a = np.asarray(Image.open(os.path.join(P.ROOT, 'assets/source/head_reference/sw_side.png')).convert('L')).astype(np.float32) / 255
        hp = a - ndimage.gaussian_filter(a, 4)
        _cache['buzz'] = hp / max(hp.std(), 1e-4) * 0.08
    hp = _cache['buzz']
    H, W = hp.shape
    ang = np.arctan2(Pt[:, 0], Pt[:, 2] - P.HEAD_CZ)
    u = ang * 0.09 * 3200.0
    v = -Pt[:, 1] * 3200.0
    def mt(c, n):
        k = np.floor(c / n); f = c - k * n
        return np.where(k % 2 == 0, f, n - 1 - f).astype(int)
    return -hp[mt(v, H - 1), mt(u, W - 1)]          # darker = more hair


def _hash01(Pt, cell):
    """Per-point pseudo-random value from quantised 3D position (no tiling)."""
    q = np.floor(Pt / cell).astype(np.int64)
    h = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float64) / 65535.0


def beard_top(th):
    """Upper edge of the beard: diagonal from the sideburn down to ~2 cm above the mouth corner."""
    return np.interp(th, [0, 25, 45, 62, 78, 95], [1.652, 1.656, 1.664, 1.676, 1.695, 1.700])


def _lm(k):
    return np.asarray(LM[k], float)


BROW_DY = tuple(_TW.get('brow_paint_dy', (0.0, 0.0)))


def _brow_mask(x, y, z, side):
    """Thick, straight brow (client video): inner end -> slight peak -> tail."""
    sg = 1 if side == 'l' else -1
    bi, bo = _lm(f'brow_in_{side}').copy(), _lm(f'brow_out_{side}').copy()
    bi[1] += BROW_DY[0]; bo[1] += BROW_DY[1]
    ax_ = sg * x
    t = (ax_ - abs(bi[0]) + 0.0095) / (abs(bo[0]) - abs(bi[0]) + 0.0115)
    yc = bi[1] + (bo[1] - bi[1]) * t + 0.0018 * np.sin(np.clip(t, 0, 1) * np.pi)
    half = np.interp(t, [-0.05, 0.12, 0.45, 0.8, 1.02], [0.0011, 0.0021, 0.0022, 0.0018, 0.0007])
    inside = smoothstep(-0.04, 0.08, t) * smoothstep(1.08, 0.90, t)
    feather = np.where(y > yc, 0.0020, 0.0020)                      # soft top edge, crisper lower edge
    across = smoothstep(half + feather, half - 0.0008, np.abs(y - yc))
    return inside * across * smoothstep(0.03, 0.07, z - P.HEAD_CZ)


def _aniso_hash(Pt, cell, off=0.0):
    """Per-point random value on an anisotropic 3D grid (cell = (cx, cy, cz))."""
    q = np.floor((Pt + off) / np.asarray(cell)).astype(np.int64)
    h = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
    h = (h ^ (h >> 13)) * 1274126177
    return ((h ^ (h >> 16)) & 0xFFFF).astype(np.float64) / 65535.0


def value_noise(Pt, cell, seed=0.0):
    """Smooth 3D value noise in [0, 1] (trilinear over a hashed lattice): blotchy
    skin colour without grid artefacts."""
    g = Pt / cell + seed
    i0 = np.floor(g); f = g - i0; f = f * f * (3 - 2 * f)
    out = 0.0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                q = (i0 + [dx, dy, dz]).astype(np.int64)
                h = (q[:, 0] * 73856093) ^ (q[:, 1] * 19349663) ^ (q[:, 2] * 83492791)
                h = (h ^ (h >> 13)) * 1274126177
                v = ((h ^ (h >> 16)) & 0xFFFF).astype(np.float64) / 65535.0
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + v * w
    return out


def stubble(Pt, density, white, dark, light, cell=0.00042, layers=3, stretch=(1.0, 1.9, 1.0)):
    """Short hairs as stippled strokes: returns (colour, coverage) per point.
    density: fraction of cells holding a hair; white: chance a hair is grey/white;
    stretch: cell aspect (hairs growing down = tall cells, brows = wide cells)."""
    cov = np.zeros(len(Pt)); col = np.zeros((len(Pt), 3))
    cv = cell * np.asarray(stretch)
    for i in range(layers):
        off = 0.000173 * (i + 1) * np.array([1.0, 0.61, 1.37])
        r_on = _aniso_hash(Pt, cv, off)
        r_c = _aniso_hash(Pt, cv, off + 0.0101)
        on = (r_on < density / layers * 1.6).astype(float)
        hc = np.where((r_c < white)[:, None], light[None, :] * (0.93 + 0.14 * r_c[:, None]), dark[None, :] * (0.93 + 0.14 * r_c[:, None]))
        new = on * (1 - cov)
        col += hc * new[:, None]; cov += new
    col = col / np.maximum(cov, 1e-6)[:, None]
    return col, cov


_EAR_TREE = {}


_EAR_ELL = {}


def ear_mask(Pt, margin=0.006, feather=0.004):
    """Smooth arch around each ear (ellipse in the side plane fitted to the
    MakeHuman ear vertices, plus a margin): 1 on/near the ear, 0 away."""
    if 'e' not in _EAR_ELL:
        d = np.load(os.path.join(P.ROOT, 'build/cache/mh_head.npz'))
        inv = {int(b): i for i, b in enumerate(d['base_idx'])}
        for side, sg in (('l', 1), ('r', -1)):
            idx = []
            for l in open(os.path.join(P.ROOT, 'build/makehuman/targets', f'ears_{side}-ear-wing-incr.target')):
                if l.strip() and not l.startswith('#'):
                    t = l.split()
                    if np.linalg.norm([float(t[1]), float(t[2]), float(t[3])]) > 0.002 and int(t[0]) in inv:
                        idx.append(inv[int(t[0])])
            q = d['V'][np.unique(idx)]
            c = np.array([(q[:, 1].min() + q[:, 1].max()) / 2, (q[:, 2].min() + q[:, 2].max()) / 2])
            ax_ = np.array([(q[:, 1].max() - q[:, 1].min()) / 2, (q[:, 2].max() - q[:, 2].min()) / 2])
            _EAR_ELL[sg] = (c, ax_, np.abs(q[:, 0]).min())
        _EAR_ELL['e'] = True
    out = np.zeros(len(Pt))
    for sg in (1, -1):
        c, ax_, xmin = _EAR_ELL[sg]
        r = np.sqrt(((Pt[:, 1] - c[0]) / (ax_[0] + margin)) ** 2 + ((Pt[:, 2] - c[1]) / (ax_[1] + margin)) ** 2)
        rr = 1 + feather / (min(ax_) + margin)
        side = smoothstep(xmin - 0.02, xmin - 0.012, Pt[:, 0] * sg)
        out = np.maximum(out, smoothstep(rr, 1.0, r) * side)
    return out


def ear_distance(Pt):
    """Distance to the nearest ear vertex (the vertices MakeHuman's ear-wing
    targets move), so hair paint and the hair shell keep a skin margin."""
    if 't' not in _EAR_TREE:
        from scipy.spatial import cKDTree
        d = np.load(os.path.join(P.ROOT, 'build/cache/mh_head.npz'))
        inv = {int(b): i for i, b in enumerate(d['base_idx'])}
        idx = []
        for side in ('l', 'r'):
            f = os.path.join(P.ROOT, 'build/makehuman/targets', f'ears_{side}-ear-wing-incr.target')
            for l in open(f):
                if l.strip() and not l.startswith('#'):
                    t = l.split()
                    if np.linalg.norm([float(t[1]), float(t[2]), float(t[3])]) > 0.002 and int(t[0]) in inv:
                        idx.append(inv[int(t[0])])
        _EAR_TREE['t'] = cKDTree(d['V'][np.unique(idx)])
    return _EAR_TREE['t'].query(Pt)[0]


def clipped_density(Pt):
    """Where the clipper-short hair grows (client video): the temples with a
    near-vertical front edge ~33 mm behind the outer eye corner, a narrow
    sideburn to tragus level, above and behind the ears, the back down to the
    nape. A ~5 mm skin margin around the ears."""
    x, y, z = Pt.T
    ax = np.abs(x)
    th = np.degrees(np.abs(np.arctan2(x, z - P.HEAD_CZ)))
    ear = ear_mask(Pt, margin=0.003)
    # temples + sideburn: front edge by height (z of the outer canthus is ~0.125)
    zf = np.interp(y, [1.660, 1.685, 1.700, 1.770], [0.046, 0.068, 0.092, 0.092])
    zf = zf + 0.002 * (value_noise(Pt, 0.004, 8.8) - 0.5) * 2
    zone_b = smoothstep(zf + 0.007, zf - 0.007, z) * smoothstep(0.042, 0.052, ax) * smoothstep(1.645, 1.653, y)
    # upper temple corners next to the forehead
    bc = np.interp(th, [30, 38, 45, 52], [1.762, 1.745, 1.728, 1.712])
    zone_c = smoothstep(28, 36, th) * smoothstep(bc - 0.003, bc + 0.006, y)
    # above/behind the ears and the back, down to the nape
    ba = np.interp(th, [80, 100, 110, 130, 180], [1.662, 1.650, 1.640, 1.612, 1.600])
    zone_a = smoothstep(70, 80, th) * smoothstep(ba - 0.003, ba + 0.006, y)
    dens = np.maximum(np.maximum(zone_a, zone_b), zone_c)
    return np.clip(dens * (1 - ear), 0, 1)


SIDE_DARK = np.array([0.26, 0.21, 0.17]); SIDE_LIGHT = np.array([0.58, 0.55, 0.52])


def clipped_hair(Pt, base, dens):
    """Clipper-cut sides: a soft grey-brown tone (short dense hair) with fine
    lighter/darker hair texture, scalp barely showing."""
    tone = np.array([0.48, 0.39, 0.32])[None, :] * (0.92 + 0.16 * value_noise(Pt, 0.006, 3.1)[:, None])
    cover = 0.85 * smoothstep(0.0, 0.6, dens)                            # ~15 % scalp showing toward the edges
    shade = base * (1 - cover[:, None]) + tone * cover[:, None]
    col, cov = stubble(Pt, 0.9 * dens, 0.35, SIDE_DARK, SIDE_LIGHT, cell=0.00036, layers=3, stretch=(1.0, 2.5, 1.0))
    k = np.clip(cov * 0.45, 0, 1)[:, None]
    return shade * (1 - k) + col * k


def _lip_masks(Pt):
    x, y, z = Pt.T
    st, ml = _lm('stomion'), _lm('mouth_l')
    hw = abs(ml[0]) + 0.0005
    u = np.clip(np.abs(x) / hw, 0, 1.5)
    dy = y - st[1]
    hu = 0.0050 * np.clip(1 - u, 0, 1) ** 1.1                      # upper lip red tapers to the corners
    upper = smoothstep(hu + 0.0007, hu - 0.0007, dy) * smoothstep(-0.0004, 0.0004, dy)
    hl = 0.0042 * (1 - u ** 2)
    lower = smoothstep(-hl - 0.0009, -hl + 0.0009, dy) * smoothstep(0.0004, -0.0004, dy)
    frontal = smoothstep(0.05, 0.09, z - P.HEAD_CZ)
    lips = np.clip((upper + lower) * (u < 1.0) * frontal, 0, 1)
    return lips, upper * (u < 1.0) * frontal, lower * (u < 1.0) * frontal, dy, u, frontal


BEARD_ZONES = {}


def beard_masks(Pt, N):
    """Stubble zones (client video): dark moustache from the nostril sill, dark
    soul patch, near-white chin, darkest under the jaw, sparse soft upper border
    on the cheeks, clean neck ~1.5 cm below the jaw edge.
    Returns (density, white fraction, moustache mask)."""
    x, y, z = Pt.T
    ax = np.abs(x)
    th = np.degrees(np.abs(np.arctan2(x, z - P.HEAD_CZ)))
    st, sub = _lm('stomion'), _lm('subnasale')
    lips, upper, lower, dy, u, frontal = _lip_masks(Pt)
    bt = beard_top(th)
    beard = smoothstep(bt + 0.004, bt - 0.014, y) * (0.85 + 0.15 * value_noise(Pt, 0.004, 2.2))
    sill = min(sub[1], _lm('ala_l')[1] - 0.001)                         # hairs start right under the nostril sill
    mous = (smoothstep(sill + 0.0015, sill - 0.0015, y) * smoothstep(0.050, 0.008, ax) * frontal
            * (0.7 + 0.3 * value_noise(Pt, 0.003, 7.7))
            * smoothstep(st[1] + 0.0005, st[1] + 0.003, y))
    if N is not None:
        mous *= smoothstep(-0.55, -0.25, N[:, 1])                          # not inside the nostrils
    beard = np.maximum(beard, mous)
    # soul patch 8-16 mm under the stomion (~32 mm wide, densest in the centre) + a sparser band 5-8 mm under it
    soul = (smoothstep(st[1] - 0.005, st[1] - 0.0075, y) * smoothstep(st[1] - 0.018, st[1] - 0.015, y)
            * smoothstep(0.016, 0.004, ax) * frontal)
    soul = np.maximum(soul, 0.3 * smoothstep(st[1] - 0.004, st[1] - 0.006, y) * smoothstep(st[1] - 0.009, st[1] - 0.007, y)
                      * smoothstep(0.016, 0.006, ax) * frontal)
    # mouth-corner "arms" joining the moustache to the chin
    arms = (smoothstep(0.022, 0.027, ax) * smoothstep(0.038, 0.033, ax) * smoothstep(st[1] + 0.005, st[1] + 0.002, y)
            * smoothstep(st[1] - 0.014, st[1] - 0.010, y) * frontal)
    beard = np.maximum(beard, np.maximum(soul, arms))
    bb = np.interp(th, [0, 40, 70, 95], [1.560, 1.566, 1.588, 1.610])
    beard *= smoothstep(bb - 0.006, bb, y)
    beard *= np.where(th > 80, smoothstep(102, 88, th), 1.0)
    beard *= 1 - np.clip(lips * 1.4, 0, 1)
    beard *= smoothstep(0.004, 0.010, ear_distance(Pt))                      # never on the ears
    beard *= 1 - smoothstep(0.0040, 0.0, -dy - 0.0050 * (1 - np.clip(u, 0, 1) ** 2)) * (dy < 0) * frontal * (u < 1)  # 2.5 mm bare under the lip
    hu = 0.0050 * np.clip(1 - u, 0, 1) ** 1.1
    beard *= 1 - smoothstep(hu + 0.0015, hu + 0.0005, dy) * (dy > 0) * frontal * (u < 1)                          # ~1 mm bare above the lip
    # white chin: soft, irregular borders (no hard rectangle)
    wob = value_noise(Pt, 0.005, 6.6) - 0.5
    chin = smoothstep(st[1] - 0.014, st[1] - 0.032, y + 0.004 * wob) * smoothstep(0.060 + 0.010 * wob, 0.012, ax)
    under = smoothstep(-0.25, -0.55, N[:, 1]) if N is not None else 0.0
    white = 0.15 + 0.08 * smoothstep(45, 75, th)
    white = white * (1 - mous) + 0.12 * mous
    white = white * (1 - soul) + 0.18 * soul
    white = white * (1 - arms) + 0.18 * arms
    white = white * (1 - chin) + 0.40 * chin
    white = white * (1 - under * (1 - chin)) + 0.12 * under * (1 - chin)
    dens = np.clip(beard * (0.50 + 0.40 * mous + 0.25 * soul + 0.15 * arms + 0.15 * under + 0.20 * chin) * CLIENT.get('beard', 1.0), 0, 1)
    BEARD_ZONES['soul'] = soul; BEARD_ZONES['arms'] = arms; BEARD_ZONES['chin'] = chin
    return dens, np.clip(white, 0.03, 0.8), mous


def paint_head_clean(Pt, N):
    """Clean MakeHuman-style skin painted from the client's traits (no photo):
    fair, ruddy skin, low thick straight brows, lash lines, natural lips,
    stubble strokes (white chin, dark moustache/soul patch/under-jaw), clipped
    warm-taupe sides."""
    x, y, z = Pt.T
    ax = np.abs(x)
    T = CLIENT_SKIN
    skin = np.tile(np.array(T['base']) * 0.92, (len(x), 1))
    def blob(c, r):
        return np.exp(-np.sum(((Pt - c) / r) ** 2, axis=1))
    tip, sub = _lm('tip'), _lm('subnasale')
    # ruddy: nose (strongest, pink, up the lower dorsum), broad cheeks, a blotchy flush; ears get their own red
    red = 1.2 * blob(tip + [0, 0.006, -0.006], [0.017, 0.026, 0.02])
    for sg in (-1, 1):
        red += 0.50 * blob(np.array([sg * 0.034, P.Y_EYE - 0.044, 0.090]), [0.026, 0.018, 0.05])
    floor_ = 0.22 + 0.30 * (value_noise(Pt, 0.007, 1.7) - 0.5) + 0.12 * (value_noise(Pt, 0.0022, 5.3) - 0.5)
    for side in ('l', 'r'):                                             # orbits and temples are not flushed
        floor_ = floor_ - (0.16 * np.exp(-np.sum(((Pt - _lm(f'eye_{side}')) / 0.015) ** 2, axis=1)))
    floor_ = floor_ * (1 - 0.6 * smoothstep(0.045, 0.055, ax))
    red += floor_ * (1 - 0.7 * beard_masks(Pt, N)[0])
    skin = mix(skin, np.tile(np.array(T['ruddy']) * 0.92, (len(x), 1)), np.clip(red, 0, 1.0))
    earw = np.clip(1.4 * smoothstep(0.012, 0.002, ear_distance(Pt)), 0, 1)
    skin = mix(skin, np.tile(np.array(T.get('ear', [0.70, 0.30, 0.27])), (len(x), 1)), 0.85 * earw)
    for side in ('l', 'r'):
        e = _lm(f'eye_{side}')
        cres = blob(e + [0, -0.010, 0.0], [0.016, 0.006, 0.03])
        skin *= (1 - 0.06 * cres)[:, None]                                     # under-eye crescent
        skin = mix(skin, skin * np.array([0.95, 0.94, 0.98])[None, :], cres)
    det = skin_detail(Pt, 0.55)[:, None] * 1.0 + (value_noise(Pt, 0.0012, 9.1)[:, None] - 0.5) * 0.04
    skin = np.clip(skin * (1 + det), 0, 1)
    # lips: rose-mauve, lower lip the stronger colour
    lips, upper, lower, dy, u, frontal = _lip_masks(Pt)
    skin = mix(skin, np.tile(np.array(T['lips']), (len(x), 1)), 0.85 * upper)
    skin = mix(skin, np.tile(np.array(T.get('lips_lower', T['lips'])), (len(x), 1)), 0.85 * lower)
    skin *= (1 - 0.30 * smoothstep(0.0008, 0.0, np.abs(dy)) * (u < 1.02) * frontal)[:, None]
    # nostrils: dark openings where the surface faces down under the nose
    if N is not None:
        st = _lm('stomion')
        nos = (smoothstep(-0.35, -0.75, N[:, 1]) * smoothstep(0.017, 0.010, ax) * smoothstep(0.002, 0.005, ax)
               * smoothstep(sub[1] - 0.001, sub[1] + 0.002, y) * smoothstep(tip[1] + 0.002, tip[1] - 0.003, y)
               * smoothstep(sub[2] - 0.012, sub[2] - 0.004, z))
        skin = mix(skin, np.tile(np.array([0.40, 0.16, 0.18]), (len(x), 1)), 0.35 * nos)      # soft red underside
        hole = (nos * smoothstep(-0.55, -0.85, N[:, 1]) * smoothstep(0.0030, 0.0045, ax) * smoothstep(0.013, 0.011, ax))
        skin = mix(skin, np.tile(np.array([0.24, 0.07, 0.09]), (len(x), 1)), 0.70 * hole)       # dark-red nostril openings
    # lash lines (upper only, soft) from the distance to the eyeball surface
    EV = np.load(os.path.join(P.ROOT, 'build/cache/mh_head.npz'))['EV']
    for sg in (-1, 1):
        E = EV[EV[:, 0] * sg > 0]
        c = E.mean(0); r = np.linalg.norm(E - c, axis=1).mean()
        dd = np.linalg.norm(Pt - c, axis=1) - r
        up = smoothstep(c[1] - 0.002, c[1] + 0.003, y)
        cin = abs(_lm(f'canthus_in_{"l" if sg > 0 else "r"}')[0]) if f'canthus_in_{"l" if sg > 0 else "r"}' in LM else 0.0
        outer = smoothstep(cin + 0.004, cin + 0.012, ax)
        lash = smoothstep(0.0008, 0.0002, dd) * ((0.80 + 0.20 * outer) * up + 0.06 * (1 - up)) * smoothstep(cin - 0.0015, cin - 0.0005, ax)
        skin *= (1 - 0.5 * lash)[:, None]
        k = f'canthus_in_{"l" if sg > 0 else "r"}'
        if k in LM:
            skin = mix(skin, np.tile(np.array(T['ruddy']), (len(x), 1)), 0.35 * blob(_lm(k), [0.003, 0.003, 0.004]))
    # brows: low, straight, soft, warm dark brown strokes
    brows = np.maximum(_brow_mask(x, y, z, 'l'), _brow_mask(x, y, z, 'r'))
    bcol, bcov = stubble(Pt, np.clip(brows * 1.3, 0, 1), 0.25, np.array(T['brow']), np.array(T.get('brow_light', [0.34, 0.28, 0.24])),
                         cell=0.00030, layers=4, stretch=(3.5, 1.0, 3.5))
    tint = np.clip(brows * 0.55, 0, 1)[:, None]
    skin = skin * (1 - tint) + (np.array(T['brow']) * 1.35)[None, :] * tint
    kb = np.clip(bcov * brows * 0.8 * CLIENT.get('brows', 1.0), 0, 0.40)[:, None]
    skin = skin * (1 - kb) + bcol * kb
    # stubble
    dens, white, mous = beard_masks(Pt, N)
    dz = np.clip(BEARD_ZONES['soul'] + BEARD_ZONES['arms'], 0, 1)
    light = mix(np.tile(np.array(T.get('beard_light_body', [0.64, 0.61, 0.57])), (len(x), 1)),
                np.tile(np.array(T['beard_light']), (len(x), 1)), BEARD_ZONES['chin'])
    bc, bcv = stubble(Pt, dens, white, np.array(T['beard_dark']), np.array([0.5, 0.5, 0.5]), cell=0.00040, stretch=(1.0, 4.0, 1.0))
    lightmask = (np.abs(bc - 0.5).sum(1) < 0.2)[:, None] * 1.0              # strokes drawn with the light placeholder
    jitter = bc / 0.5
    bc = bc * (1 - lightmask) + light * jitter * lightmask
    darkz = np.tile(np.array(T.get('beard_darkzone', [0.30, 0.27, 0.24])), (len(x), 1))
    mz = np.tile(np.array([0.30, 0.26, 0.23]), (len(x), 1))
    bc = bc * (1 - lightmask) * (1 - dz[:, None]) + darkz * (1 - lightmask) * dz[:, None] + bc * lightmask
    bc = bc * (1 - (1 - lightmask) * mous[:, None]) + mz * (1 - lightmask) * mous[:, None]
    tone = mix(np.tile(np.array([0.52, 0.48, 0.44]), (len(x), 1)), np.tile(np.array([0.66, 0.62, 0.58]), (len(x), 1)), white)
    shadow = np.clip(dens * 0.55, 0, 1)[:, None]
    skin = skin * (1 - shadow) + tone * shadow
    k = np.clip(bcv * (0.40 + 0.30 * mous + 0.12 * BEARD_ZONES['arms']), 0, 1)[:, None]
    skin = skin * (1 - k) + bc * k
    skin *= (1 - 0.30 * mous * dens)[:, None]                         # moustache: the darkest front zone
    # clipped sides/back, temples, sideburns
    dside = clipped_density(Pt)
    scalp = clipped_hair(Pt, skin, dside)
    top = smoothstep(P.hair_line(Pt) - 0.006, P.hair_line(Pt) + 0.001, y)
    out = mix(scalp, paint_hair(Pt, N), top)
    return np.clip(out, 0, 1)


def ao_floor(Pt):
    """Minimum ambient occlusion on the head: none of the baked AO darkens the
    lids/orbits (no eyeshadow) or the lips (no double lip line)."""
    f = np.zeros(len(Pt))
    for side in ('l', 'r'):
        f = np.maximum(f, 0.85 * np.exp(-np.sum(((Pt - _lm(f'eye_{side}')) / [0.022, 0.016, 0.03]) ** 2, axis=1)))
    st = _lm('stomion')
    f = np.maximum(f, 0.90 * np.exp(-np.sum(((Pt - st) / [0.030, 0.010, 0.03]) ** 2, axis=1)))
    return f


def head_roughness(Pt, N):
    """ORM-style map for the head (G = roughness): matte hair zones (stubble,
    brows, clipped and long hair), satin skin, slightly glossier nose and lips."""
    x, y, z = Pt.T
    dens, _, _ = beard_masks(Pt, N)
    brows = np.maximum(_brow_mask(x, y, z, 'l'), _brow_mask(x, y, z, 'r'))
    hair = np.maximum(brows, clipped_density(Pt))
    top = smoothstep(P.hair_line(Pt) - 0.004, P.hair_line(Pt) + 0.006, y)
    lips = _lip_masks(Pt)[0]
    tip = _lm('tip')
    nose = np.exp(-np.sum(((Pt - (tip + [0, -0.001, -0.003])) / [0.012, 0.011, 0.015]) ** 2, axis=1))
    r = 0.60 - 0.20 * nose - 0.12 * lips
    r = r * (1 - dens) + 0.72 * dens
    r = r * (1 - hair) + 0.86 * hair
    r = r * (1 - top) + 0.55 * top
    out = np.zeros((len(x), 3)); out[:, 0] = 1.0; out[:, 1] = np.clip(r, 0, 1); out[:, 2] = 0.0
    return out


CLIENT_SKIN = dict(base=[0.80, 0.60, 0.50], ruddy=[0.80, 0.45, 0.40], lips=[0.66, 0.42, 0.40],
                   brow=[0.16, 0.12, 0.10], beard_light=[0.74, 0.72, 0.70], beard_dark=[0.22, 0.18, 0.16])
CLIENT_SKIN.update(_TW.get('skin', {}))


def paint_head(Pt, N):
    x, y, z = Pt.T
    img = face_photo()
    xu = _unwrap_x(x)
    pp = face_warp(xu, y)
    face = sample(img, pp[:, 0], pp[:, 1])
    th = np.degrees(np.abs(np.arctan2(x, z - P.HEAD_CZ)))
    ang = np.degrees(np.arctan2(x, z - P.HEAD_CZ))
    ear = np.zeros(len(x))
    for side in ('L', 'R'):
        c, h_, d_ = ear_info(side)
        e = ((x - c[0]) / 0.020) ** 2 + ((y - c[1]) / (h_ * 0.62)) ** 2 + ((z - c[2]) / (d_ * 0.70)) ** 2
        ear = np.maximum(ear, smoothstep(1.25, 0.85, e))
    # photo only on the face; wide feather; fade before the surface turns away (no stretched smears)
    ocx, ocy, oax, oay = CLIENT['oval']
    ex = (pp[:, 0] - ocx) / oax
    ey = (pp[:, 1] - ocy) / oay
    oval = smoothstep(1.0, 0.72, np.sqrt(ex * ex + ey * ey))
    f0, f1 = CLIENT['forehead']
    w_face = oval * smoothstep(f0, f1, pp[:, 1]) * smoothstep(60, 44, th)
    # synthesized skin: the photo's own average tone, pores, ruddy ears
    tone = skin_tone()
    detail = skin_detail(Pt)[:, None]
    skin = np.clip(tone[None, :] * (1 + detail * 2.0), 0, 1)
    skin = skin * (1 - ear[:, None]) + skin * np.array([1.06, 0.90, 0.88])[None, :] * ear[:, None]
    # salt-and-pepper beard over jaw, lower cheeks and chin (fading on the neck)
    rnd = _hash01(Pt, 0.0005); rnd2 = _hash01(Pt + 0.37, 0.0007)
    bt = beard_top(th)
    beard = smoothstep(bt + 0.004, bt - 0.006, y) * smoothstep(1.548, 1.575, y) * smoothstep(-0.035, -0.005, z - P.HEAD_CZ + 0.03)
    beard *= np.where(th > 80, smoothstep(98, 82, th), 1.0)
    grey = smoothstep(32, 12, th) * smoothstep(1.625, 1.60, y)          # grey/white chin
    wf = np.clip(0.35 + 0.35 * grey, 0, 1)[:, None]
    hair_b = np.array([[0.76, 0.74, 0.72]]) * wf + np.array([[0.24, 0.20, 0.18]]) * (1 - wf)
    kb = np.clip(beard * (0.40 + 0.25 * rnd) * CLIENT.get('beard', 1.0), 0, 0.7)[:, None]
    skin = skin * (1 - kb) + hair_b * kb
    # clipped sides/back (client video): dense grey-brown hair down to the ears and the nape,
    # temple hair framing the forehead down to brow-tail height, sideburns into the beard
    sides = smoothstep(46, 58, th) * smoothstep(1.595, 1.625, y)
    temple = smoothstep(38, 50, th) * smoothstep(1.712, 1.726, y)
    burns = smoothstep(56, 64, th) * smoothstep(84, 76, th) * smoothstep(1.650, 1.662, y)
    dens = np.clip(np.maximum(np.maximum(sides, temple), burns) * (1 - ear * 0.97), 0, 1)
    pepper = (rnd2 < 0.28)
    side_c = np.array([[0.36, 0.30, 0.26]]) * 0.72 + np.array([[0.58, 0.56, 0.54]]) * 0.28
    side_c = side_c * (0.9 + 0.2 * rnd[:, None])
    k = np.clip(dens * (0.78 + 0.08 * rnd), 0, 0.88)[:, None]
    scalp = skin * (1 - k) + side_c * k
    # long top hair painted on the scalp too (matches the hair shell above it)
    top = smoothstep(P.hair_line(Pt) - 0.004, P.hair_line(Pt) + 0.006, y)
    scalp = mix(scalp, paint_hair(Pt, N), top)
    out = mix(scalp, face[:, :3], w_face)
    # soft natural hairline: fine hairs thinning out just below the long hair's front edge
    band = smoothstep(P.hair_line(Pt) - 0.009, P.hair_line(Pt) - 0.0005, y) * smoothstep(48, 30, th)
    kh = np.clip(band * (0.30 + 0.25 * rnd), 0, 0.6)[:, None]
    return out * (1 - kh) + np.array([0.30, 0.25, 0.21])[None, :] * kh


def _photo_y_of_row(r):
    ys = np.array([LM['glabella'][1], P.Y_EYE, LM['subnasale'][1], LM['stomion'][1], LM['menton'][1]])
    rows = np.array(_ROWS)
    if r > rows[-1]:
        return ys.min() - (r - rows[-1]) / FACE_S
    return float(np.interp(r, rows, ys))


_EARS = {}


def ear_info(side):
    if side not in _EARS:
        sg = 1 if side == 'L' else -1
        # the ear is the part that sticks out beyond the skull's side
        xmax = np.abs(_mhv[:, 0]).max()
        m = (_mhv[:, 0] * sg > xmax - 0.011) & (_mhv[:, 1] > 1.60) & (_mhv[:, 1] < 1.78)
        q = _mhv[m]
        c = np.array([(q[:, 0].min() + q[:, 0].max()) / 2, (q[:, 1].min() + q[:, 1].max()) / 2, (q[:, 2].min() + q[:, 2].max()) / 2])
        _EARS[side] = (c, q[:, 1].max() - q[:, 1].min(), q[:, 2].max() - q[:, 2].min())
    return _EARS[side]


def paint_ear_mh(Pt, side):
    c, h, d = ear_info(side)
    sg = 1 if side == 'L' else -1
    name = 'ear_b' if side == 'L' else 'ear_a'
    img = piece(name)
    bx0, by0, bx1, by1 = META[name]['bbox']
    s = (by1 - by0) / (h * 1.02)
    # viewed from outside: left ear -> face (+z) on the image left; right ear mirrored
    px = (bx0 + bx1) / 2 - sg * (Pt[:, 2] - c[2]) * s
    py = (by0 + by1) / 2 - (Pt[:, 1] - c[1]) * s
    col = sample(img, px, py)
    inside = np.clip(col[:, 3] * 1.4, 0, 1)
    w = inside * smoothstep(0.060, 0.072, Pt[:, 0] * sg) * smoothstep(0.016, 0.004, np.abs(Pt[:, 1] - c[1]) - h / 2)
    return col[:, :3], w


def ear_frame():
    return P.EAR_CENTER, P.EAR_N, P.EAR_UP, P.EAR_BACK


def paint_ear(Pt, N, side='L'):
    q = Pt.copy()
    if side == 'R':
        q[:, 0] *= -1
    c, n, up, back = ear_frame()
    d = q - c
    lz, ly = d @ back, d @ up
    name = 'ear_b' if side == 'L' else 'ear_a'
    img = piece(name)
    bx0, by0, bx1, by1 = META[name]['bbox']
    cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
    s = (by1 - by0) / 0.064
    sgn = 1 if side == 'L' else -1
    px = cx + sgn * (lz - 0.004) * s
    py = cy - (ly + 0.001) * s
    col = sample(img, px, py)
    # back of the ear / attachment blends to plain skin
    return mix(np.tile(skin_tone(), (len(px), 1)), col[:, :3], np.clip(col[:, 3] * 1.3, 0, 1) * 0.9 + 0.1)


# ====================================================================== HAIR
HAIR_B = P.BUN_CENTER - P.BUN_AXIS * 0.030      # where the slicked hair gathers (bun seated on the MakeHuman scalp)


def hair_px(Pt):
    """Pixel coords in the slicked-hair swatch: strands converge on the bun, so
    map polar coordinates around the gather point onto the swatch's dome."""
    bx0, by0, bx1, by1 = META['hair_top']['bbox']
    cx = (bx0 + bx1) / 2
    top = by0 + 2
    R = (by1 - by0) - 4
    d = Pt - HAIR_B
    fwd = np.array([0, -0.25, 1.0]); fwd /= np.linalg.norm(fwd)
    side = np.array([1.0, 0, 0])
    a = np.arctan2(d @ side, d @ fwd)                      # 0 = toward forehead
    dist = np.linalg.norm(d, axis=1)
    rmax = np.interp(np.abs(a), [0, 0.6, 1.2, 1.6, 2.2, 3.15], [0.150, 0.140, 0.112, 0.090, 0.065, 0.055])
    t = np.clip(dist / rmax, 0, 1) * 0.90          # stay inside the swatch's dome (its rim is dark)
    ai = np.clip(a / (np.pi * 0.62), -1, 1) * (np.pi / 2) * 0.93
    px = cx + np.sin(ai) * t * (bx1 - bx0) / 2
    py = top + np.cos(ai) * t * R
    return px, py, a


def bun_px(Pt):
    bx0, by0, bx1, by1 = META['bun']['bbox']
    cx, cy, rad = (bx0 + bx1) / 2, (by0 + by1) / 2, (bx1 - bx0) / 2
    q = Pt - P.BUN_CENTER
    h = q @ P.BUN_AXIS
    radial = q - h[:, None] * P.BUN_AXIS
    ref = np.cross(P.BUN_AXIS, [1.0, 0, 0]); ref /= np.linalg.norm(ref)
    ref2 = np.cross(P.BUN_AXIS, ref)
    ang = np.arctan2(radial @ ref2, radial @ ref)
    r = np.linalg.norm(radial, axis=1) / 0.036
    r = np.where(h < 0, 1.0 - 0.15 * (h / 0.024), r)
    r = np.clip(r, 0, 1) * 0.96
    return cx + np.cos(ang) * r * rad, cy + np.sin(ang) * r * rad, ang


HAIR_ATLAS = (1024, 256)
HAIR_SLOT = {'hair_top': (0, 0, 2.0), 'bun': (460, 0, 2.0)}      # x, y offset and upscale in the atlas


def hair_atlas(path):
    """Hair swatch + bun swatch, upscaled side by side (direct UVs, no re-bake)."""
    from PIL import Image
    W, H = HAIR_ATLAS
    out = np.zeros((H, W, 3), np.float32)
    for name, (ox, oy, sc) in HAIR_SLOT.items():
        img = piece(name)[..., :3]
        im = Image.fromarray((img * 255).astype(np.uint8)).resize((int(img.shape[1] * sc), int(img.shape[0] * sc)), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32) / 255
        out[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a
    # pad the rest with the swatch's mean colour
    mean = piece('hair_top')[..., :3].reshape(-1, 3).mean(0)
    empty = out.sum(2) == 0
    out[empty] = mean
    from texlib import save_rgb
    save_rgb(path, out, quality=94)
    return out


def hair_uv(Pt, which):
    if which == 'hair_top':
        px, py, a = hair_px(Pt)
    else:
        px, py, a = bun_px(Pt)
    ox, oy, sc = HAIR_SLOT[which]
    W, H = HAIR_ATLAS
    return np.c_[(ox + px * sc) / W, 1 - (oy + py * sc) / H], a


_STRANDS = np.random.default_rng(11).random(4096)


def paint_hair(Pt, N):
    """Long top hair combed flat straight back into the bun (client's video):
    fine strands converging on the bun, dark ash brown with sparse grey; below
    the hair line the same shell carries the clipper-short sides."""
    d = Pt - HAIR_B
    fwd = np.array([0, -0.25, 1.0]); fwd /= np.linalg.norm(fwd)
    a = np.arctan2(d[:, 0], d @ fwd)
    dist = np.linalg.norm(d, axis=1)
    lat = a * np.maximum(dist, 0.03) + 0.003 * np.sin(dist * 45.0 + a * 4.0)      # metres across the flow
    def band(w, salt):
        f = lat / w + 1000.3 + salt
        i0 = np.floor(f).astype(int); t = f - i0; t = t * t * (3 - 2 * t)
        return _STRANDS[(i0 * 7 + int(salt)) % 4096] * (1 - t) + _STRANDS[((i0 + 1) * 7 + int(salt)) % 4096] * t
    r = band(0.0008, 0); r2 = band(0.00035, 17); clump = band(0.007, 101); clump2 = band(0.0031, 53)
    base = np.array([0.15, 0.115, 0.09]); dark = np.array([0.08, 0.065, 0.05]); grey = np.array([0.52, 0.49, 0.45])
    along = 0.5 + 0.5 * np.sin(dist * 140.0 + r * 6.0)
    col = base[None, :] * (0.92 + 0.16 * r[:, None]) * (0.90 + 0.20 * clump2[:, None]) * (0.94 + 0.12 * along[:, None])
    col = mix(col, dark[None, :].repeat(len(r), 0), 0.18 * (1 - clump2) ** 2)
    gthr = 0.84                                           # fine, evenly scattered grey strands
    g = smoothstep(gthr - 0.04, gthr + 0.04, r2)
    col = mix(col, grey[None, :].repeat(len(r), 0), 0.22 * g)
    above = Pt[:, 1] - P.hair_line(Pt)
    dens = np.maximum(clipped_density(Pt), 0.6)
    side = clipped_hair(Pt, np.tile(np.array([0.46, 0.39, 0.34]), (len(Pt), 1)), dens)
    return mix(side, col, smoothstep(-0.004, 0.003, above))


def hair_roughness(Pt, N):
    """G = roughness for the hair shell: matte clipped sides, satin slicked top."""
    above = Pt[:, 1] - P.hair_line(Pt)
    top = smoothstep(-0.004, 0.003, above)
    r = 0.85 * (1 - top) + 0.45 * top
    out = np.zeros((len(Pt), 3)); out[:, 0] = 1.0; out[:, 1] = r
    return out


def bun_roughness(Pt, N):
    out = np.zeros((len(Pt), 3)); out[:, 0] = 1.0; out[:, 1] = 0.6
    return out


def paint_bun(Pt, N):
    px, py, a = bun_px(Pt)
    c = sample(piece('bun'), px, py)[:, :3]
    lum = c.mean(1, keepdims=True)
    # re-tint the swatch to the client's ash brown
    out = lum / max(float(piece('bun')[..., :3].mean()), 1e-3) * np.array([0.30, 0.27, 0.245])[None, :]
    tie = P.tie_sdf(Pt) < 0.0012
    out[tie] = np.array([0.03, 0.03, 0.032])
    return np.clip(out, 0, 1)


# ====================================================================== ARMS / HANDS
def paint_arm(Pt, N, side='L'):
    q = Pt.copy(); n = N.copy()
    if side == 'R':
        q[:, 0] *= -1; n[:, 0] *= -1
    el = P.HEAD['lowerarm_l']; wr = P.HEAD['hand_l']
    L = wr[0] - el[0]
    # ---- forearm (cylindrical, dorsal = +Y in the T-pose)
    phi = np.degrees(np.arctan2(q[:, 2] - el[2], q[:, 1] - el[1]))     # 0 = up(dorsal), +90 = front (thumb side)
    v = np.clip((q[:, 0] - (el[0] - 0.03)) / (L + 0.03), 0, 1)
    fa, fb = piece('forearm_a'), piece('forearm_b')
    ba, bb = META['forearm_a']['bbox'], META['forearm_b']['bbox']
    def fore(img, bbox, ph):
        x0, y0, x1, y1 = bbox
        return sample(img, (x0 + x1) / 2 + (ph / 90.0) * (x1 - x0) / 2 * 0.92, y0 + 4 + v * (y1 - y0 - 8))
    dors = fore(fa, ba, np.clip(phi, -90, 90) * (1 if side == 'L' else -1))
    ph2 = np.where(phi > 0, 180 - phi, -180 - phi)
    vent = fore(fb, bb, np.clip(ph2, -90, 90) * (1 if side == 'L' else -1))
    wv = smoothstep(80, 100, np.abs(phi))
    forearm = mix(dors[:, :3], vent[:, :3], wv)
    # ---- hand (planar from above / below)
    hs = 'l' if side == 'L' else 'r'
    dn, pn = f'hand_dorsal_{hs}', f'hand_palm_{hs}'
    dimg, pimg = piece(dn), piece(pn)
    tip_x = P.TAIL['middle_03_l'][0] + 0.004
    wx = wr[0] - 0.004
    def hand(img, name, palm):
        x0, y0, x1, y1 = META[name]['bbox']
        hl = (y1 - y0)
        s = hl / (tip_x - wx)
        cz = -0.064                     # hand centre line (middle finger) in z
        cxi = (x0 + x1) / 2 + (2 if not palm else 0)
        # +z (thumb) is image-right on the dorsal view of the left hand, image-left on its palm view
        sgn = 1 if (not palm) else -1
        if side == 'R':
            sgn = -sgn
        px = cxi + sgn * (q[:, 2] - cz) * s * 1.05
        py = y1 - (q[:, 0] - wx) * s
        return sample(img, px, py)
    dcol = hand(dimg, dn, False)
    pcol = hand(pimg, pn, True)
    wp = smoothstep(0.25, -0.25, n[:, 1])
    handc = mix(dcol[:, :3], pcol[:, :3], wp)
    wh = smoothstep(wr[0] - 0.012, wr[0] + 0.012, q[:, 0])
    return mix(forearm, handc, wh)


# ====================================================================== T-SHIRT
_tee_ref = None


def cross_sections(V, y_bins, xlim=0.2):
    zc, a, b = [], [], []
    for y in y_bins:
        m = (np.abs(V[:, 1] - y) < 0.012) & (np.abs(V[:, 0]) < xlim)
        if m.sum() < 8:
            zc.append(np.nan); a.append(np.nan); b.append(np.nan); continue
        zc.append((V[m, 2].max() + V[m, 2].min()) / 2)
        a.append(np.abs(V[m, 0]).max())
        b.append((V[m, 2].max() - V[m, 2].min()) / 2)
    arr = np.array([zc, a, b])
    for r in arr:
        ok = ~np.isnan(r)
        r[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], r[ok])
    return arr


def set_tee_ref(V):
    global _tee_ref
    ys = np.linspace(0.94, 1.52, 40)
    zc, a, b = cross_sections(V, ys, xlim=0.17)
    _tee_ref = (ys, zc, a, b)


def tee_without_collar(name, ell):
    """The flat-lay tee photos include their own crew collar; paint it out with
    the surrounding fabric so it doesn't print a second collar on the chest."""
    key = 'nocollar_' + name
    if key not in _cache:
        img = piece(name).copy()
        H, W = img.shape[:2]
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cx, cy, ax, ay = ell
        hole = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2 < 1.0
        rgba = img.copy(); rgba[..., 3] = np.where(hole, 0, img[..., 3])
        from texlib import dilate_colors
        filled = dilate_colors(rgba, iters=80)
        filled = np.stack([ndimage.gaussian_filter(filled[..., c], 3) for c in range(3)], -1)
        img[..., :3] = np.where(hole[..., None], filled, img[..., :3])
        img[..., 3] = np.maximum(img[..., 3], hole.astype(np.float32))
        _cache[key] = img
    return _cache[key]


def paint_tee(Pt, N):
    ys, zc_t, a_t, b_t = _tee_ref
    y = Pt[:, 1]
    zc = np.interp(y, ys, zc_t); a = np.interp(y, ys, a_t); b = np.interp(y, ys, b_t)
    t = np.arctan2(Pt[:, 0] / a, (Pt[:, 2] - zc) / b)
    s = ellipse_arc(t, a, b)
    Q = ellipse_arc(np.full(len(t), np.pi / 2), a, b)
    u = s / Q                                                          # -1..1 front, beyond = back
    out = np.zeros((len(y), 3))
    row_of = lambda yy, top, bot: top + (1.505 - yy) / (1.505 - P.TEE_HEM) * (bot - top)
    fimg, bimg = tee_without_collar('tee_front', (160, 20, 66, 34)), tee_without_collar('tee_back', (156, 14, 66, 22))
    # front: wearer's left (+x) is image right
    f = sample(fimg, 160 + u * 143, row_of(y, 8, 330))
    ub = np.where(u > 0, 2 - u, -2 - u)                                # -1..1 across the back
    bk = sample(bimg, 156 - ub * 143, row_of(y, 7, 326))
    w = smoothstep(0.92, 1.08, np.abs(u))
    return mix(f[:, :3], bk[:, :3], w)


# ====================================================================== JEANS
_jeans_ref = None


def set_jeans_ref(V):
    global _jeans_ref
    ys = np.linspace(0.08, 1.07, 60)
    pel = cross_sections(V, ys, xlim=0.25)
    legs = []
    for s in (-1, 1):
        m = V[:, 0] * s > 0.004
        legs.append(cross_sections_leg(V[m], ys, s))
    _jeans_ref = (ys, pel, legs)


def cross_sections_leg(V, ys, s):
    cx, cz, a, b = [], [], [], []
    for y in ys:
        m = np.abs(V[:, 1] - y) < 0.012
        if m.sum() < 8:
            cx.append(np.nan); cz.append(np.nan); a.append(np.nan); b.append(np.nan); continue
        cx.append((V[m, 0].max() + V[m, 0].min()) / 2); cz.append((V[m, 2].max() + V[m, 2].min()) / 2)
        a.append((V[m, 0].max() - V[m, 0].min()) / 2); b.append((V[m, 2].max() - V[m, 2].min()) / 2)
    arr = np.array([cx, cz, a, b])
    for r in arr:
        ok = ~np.isnan(r)
        r[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], r[ok])
    return arr


JEANS_CROTCH = 0.865


def _segments(mask_row):
    xs = np.nonzero(mask_row)[0]
    if len(xs) == 0:
        return []
    br = np.nonzero(np.diff(xs) > 1)[0]
    starts = np.r_[xs[0], xs[br + 1]]; ends = np.r_[xs[br], xs[-1]]
    return list(zip(starts, ends))


def _leg_tables(name, split_row):
    """Per-row centre/half-width of each leg (left/right in image) below the crotch,
    and of the whole garment above."""
    a = piece(name)[..., 3] > 0.5
    H = a.shape[0]
    whole = np.zeros((H, 2)); legL = np.zeros((H, 2)); legR = np.zeros((H, 2))
    for r in range(H):
        segs = _segments(a[r])
        if not segs:
            continue
        lo, hi = segs[0][0], segs[-1][1]
        whole[r] = ((lo + hi) / 2, (hi - lo) / 2)
        if len(segs) >= 2 and r >= split_row:
            l, rr = segs[0], segs[-1]
            legL[r] = ((l[0] + l[1]) / 2, (l[1] - l[0]) / 2)
            legR[r] = ((rr[0] + rr[1]) / 2, (rr[1] - rr[0]) / 2)
    # fill rows without a split from the nearest split row
    for T in (legL, legR):
        ok = T[:, 1] > 0
        idx = np.nonzero(ok)[0]
        for c in range(2):
            T[:, c] = np.interp(np.arange(H), idx, T[idx, c])
    ok = whole[:, 1] > 0
    idx = np.nonzero(ok)[0]
    for c in range(2):
        whole[:, c] = np.interp(np.arange(H), idx, whole[idx, c])
    return whole, legL, legR


_jt = {}


def paint_jeans(Pt, N):
    ys, pel, legs = _jeans_ref
    if not _jt:
        _jt['front'] = _leg_tables('jeans_front', 140) + (8, 140, 462)
        _jt['back'] = _leg_tables('jeans_back', 180) + (8, 180, 461)
    y = Pt[:, 1]
    res = {}
    for view in ('front', 'back'):
        whole, legL, legR, top, crow, bot = _jt[view]
        img = piece('jeans_' + view)
        # rows: waist..crotch, crotch..hem
        row = np.where(y > JEANS_CROTCH,
                       top + (P.JEANS_TOP - y) / (P.JEANS_TOP - JEANS_CROTCH) * (crow - top),
                       crow + (JEANS_CROTCH - y) / (JEANS_CROTCH - P.JEANS_HEM) * (bot - crow))
        ri = np.clip(row.astype(int), 0, len(whole) - 1)
        # pelvis mapping
        zc = np.interp(y, ys, pel[0]); a = np.interp(y, ys, pel[1]); b = np.interp(y, ys, pel[2])
        t = np.arctan2(Pt[:, 0] / a, (Pt[:, 2] - zc) / b)
        s = ellipse_arc(t, a, b); Q = ellipse_arc(np.full(len(t), np.pi / 2), a, b)
        u = s / Q
        sgn = 1 if view == 'front' else -1
        uu = u if view == 'front' else np.where(u > 0, 2 - u, -2 - u)
        pelc = sample(img, whole[ri, 0] + sgn * uu * whole[ri, 1], row)
        # per-leg mapping
        legc = np.zeros_like(pelc)
        for k, s_ in enumerate((-1, 1)):
            cx, cz, la, lb = [np.interp(y, ys, legs[k][i]) for i in range(4)]
            tl = np.arctan2((Pt[:, 0] - cx) / la, (Pt[:, 2] - cz) / lb)
            sl = ellipse_arc(tl, la, lb); Ql = ellipse_arc(np.full(len(tl), np.pi / 2), la, lb)
            ul = sl / Ql
            ul = ul if view == 'front' else np.where(ul > 0, 2 - ul, -2 - ul)
            # wearer's right leg (-x) is image-left in the front view, image-right in the back view
            T = (legL if s_ < 0 else legR) if view == 'front' else (legR if s_ < 0 else legL)
            c = sample(img, T[ri, 0] + sgn * ul * T[ri, 1], row)
            m = (Pt[:, 0] * s_ >= 0)
            legc[m] = c[m]
        wl = smoothstep(JEANS_CROTCH + 0.03, JEANS_CROTCH - 0.01, y)
        res[view] = (mix(pelc[:, :3], legc[:, :3], wl), u)
    # front/back blend by which half of the local cross-section the point is on
    u = res['front'][1]
    # for legs use the leg-local angle to decide front/back
    ys_, pel_, legs_ = _jeans_ref
    ufb = np.zeros(len(y))
    for k, s_ in enumerate((-1, 1)):
        cx, cz, la, lb = [np.interp(y, ys_, legs_[k][i]) for i in range(4)]
        tl = np.abs(np.arctan2((Pt[:, 0] - cx) / la, (Pt[:, 2] - cz) / lb))
        m = Pt[:, 0] * s_ >= 0
        ufb[m] = tl[m] / (np.pi / 2)
    wl = smoothstep(JEANS_CROTCH + 0.03, JEANS_CROTCH - 0.01, y)
    fb = wl * ufb + (1 - wl) * np.abs(u)
    w = smoothstep(0.92, 1.08, fb)
    return mix(res['front'][0], res['back'][0], w)


# ====================================================================== SHOES
def paint_shoe(Pt, N, side='L'):
    q = Pt.copy(); n = N.copy()
    if side == 'R':
        q[:, 0] *= -1; n[:, 0] *= -1
    z_t, z_h = 0.212, -0.090
    L = z_t - z_h
    def side_img(name, toe_left):
        x0, y0, x1, y1 = META[name]['bbox']
        f = (z_t - q[:, 2]) / L
        px = (x0 + f * (x1 - x0)) if toe_left else (x1 - f * (x1 - x0))
        py = y1 - q[:, 1] / 0.124 * (y1 - y0)
        return sample(piece(name), px, py)[:, :3]
    outer_name, inner_name = ('shoe_side_a', 'shoe_side_b') if side == 'L' else ('shoe_side_b', 'shoe_side_a')
    outer = side_img(outer_name, side == 'L')
    inner = side_img(inner_name, side != 'L')
    w_in = smoothstep(0.15, -0.15, n[:, 0])
    col = mix(outer, inner, w_in)
    # outsole
    x0, y0, x1, y1 = META['outsole']['bbox']
    f = (z_t - q[:, 2]) / L
    sole = sample(piece('outsole'), x0 + f * (x1 - x0), (y0 + y1) / 2 + (q[:, 0] - 0.089) / 0.11 * (y1 - y0))[:, :3]
    wb = smoothstep(-0.55, -0.85, n[:, 1]) * (q[:, 1] < 0.012)
    return mix(col, sole, wb)


# ====================================================================== COAT
K_FRONT, K_BACK = 965.0, 715.0
_coat = {}


def _layout(img, segs):
    """segs: list of (src_r0, src_r1, n_out_rows, mode) -> composed image.
    mode 'copy' keeps rows (resampled to n_out_rows), 'extend' mirrors a
    flattened version of the band to fill n_out_rows."""
    from texlib import resample_rows
    out = []
    for r0, r1, n, mode in segs:
        band = img[r0:r1]
        if mode == 'copy':
            band = resample_rows(band, [0, r1 - r0], [0, n], n) if n != (r1 - r0) else band
        else:
            # lengthen plain fabric by stretching (keeps seams/edges continuous and
            # reads as vertical hanging folds), then restore crisp weave detail
            st = resample_rows(band, [0, r1 - r0], [0, n], n)
            low = np.dstack([ndimage.gaussian_filter(st[..., c], (3.0, 0.8)) for c in range(st.shape[2])])
            det = weave_detail(n, st.shape[1])
            band = low.copy()
            band[..., :3] = np.clip(low[..., :3] + det[..., None] * 0.9, 0, 1)
        out.append(band)
    return np.concatenate(out)


_weave = {}


def weave_detail(h, w):
    """High-pass cotton weave from a plain patch of the coat, mirror-tiled
    with per-tile offsets (no visible repetition at this scale)."""
    if 'hp' not in _weave:
        sw = piece('coat_back')[210:290, 70:170, :3].mean(2)
        _weave['hp'] = sw - ndimage.gaussian_filter(sw, 2.5)
    hp = _weave['hp']
    th, tw = hp.shape
    rng = np.random.default_rng(7)
    out = np.zeros((h, w))
    for y in range(0, h, th // 2):
        for x in range(0, w, tw // 2):
            oy, ox = rng.integers(0, th // 2), rng.integers(0, tw // 2)
            tile = hp[oy:oy + th // 2, ox:ox + tw // 2]
            if rng.random() < 0.5: tile = tile[::-1]
            if rng.random() < 0.5: tile = tile[:, ::-1]
            hh, ww = min(th // 2, h - y), min(tw // 2, w - x)
            out[y:y + hh, x:x + ww] = tile[:hh, :ww]
    return ndimage.gaussian_filter(out, 0.6)


def coat_textures():
    if _coat:
        return _coat
    kf, kb = K_FRONT, K_BACK
    m = lambda d: int(round(d * kf))
    y_top = 1.505
    # front panels: shoulder..chest pocket | plain | pockets | plain | hem
    seg_f = [(4, 200, 196, 'copy'), (200, 268, m(1.505 - 196 / kf - 0.99), 'extend'),
             (268, 386, 118, 'copy'), (386, 430, m(0.87 - 0.505) - 10, 'extend'), (430, 440, 10, 'copy')]
    for name in ('coat_front_l', 'coat_front_r'):
        img = piece(name)
        _coat[name] = _layout(img, [(a, b, max(n, 1), md) for a, b, n, md in seg_f])
    mb = lambda d: int(round(d * kb))
    seg_b = [(4, 300, 296, 'copy'), (300, 335, mb(1.505 - 296 / kb - 0.735), 'extend'),
             (335, 440, mb(0.735 - 0.505), 'copy')]
    _coat['coat_back'] = _layout(piece('coat_back'), seg_b)
    _coat['y_top'] = y_top
    # per-row front-edge / side-seam columns for each front panel
    for name in ('coat_front_l', 'coat_front_r'):
        a = _coat[name][..., 3] > 0.5
        H = a.shape[0]
        lo = np.array([np.nonzero(a[r])[0].min() if a[r].any() else np.nan for r in range(H)])
        hi = np.array([np.nonzero(a[r])[0].max() if a[r].any() else np.nan for r in range(H)])
        for arr in (lo, hi):
            ok = ~np.isnan(arr); arr[~ok] = np.interp(np.nonzero(~ok)[0], np.nonzero(ok)[0], arr[ok])
        _coat[name + '_lo'] = ndimage.uniform_filter1d(lo, 9)
        _coat[name + '_hi'] = ndimage.uniform_filter1d(hi, 9)
    return _coat


P.VENT_TOP = 0.735


def coat_uv(Pt):
    """Return (front_l, front_r, back) colour samples and the blend weight."""
    C = coat_textures()
    y = Pt[:, 1]
    zc, a, b = P.ref_ellipse(y)
    t = np.arctan2(Pt[:, 0] / a, (Pt[:, 2] - zc) / b)
    Q = ellipse_arc(np.full(len(t), np.pi / 2), a, b)
    s = ellipse_arc(t, a, b)
    # FRONT: side seam (|t| = 90deg) anchored to the panel's outer column; isotropic scale
    row_f = (C['y_top'] - y) * K_FRONT + 4
    rf = np.clip(row_f.astype(int), 0, len(C['coat_front_l_hi']) - 1)
    # wearer's left panel: side seam = image right edge (hi); right panel: side seam = image left edge (lo)
    side_l = np.maximum(C['coat_front_l_hi'][rf], 150)
    side_r = np.minimum(C['coat_front_r_lo'][rf], 32)
    pxl = side_l - (Q - s) * K_FRONT
    pxr = side_r + (Q + s) * K_FRONT
    fl = sample(C['coat_front_l'], pxl, row_f)
    fr = sample(C['coat_front_r'], pxr, row_f)
    front = np.where((t > 0)[:, None], fl, fr)
    # BACK: centred, wearer's left (+x) on the image left
    sb = np.where(t > 0, 2 * Q - s, -2 * Q - s)        # arc from back centre, + toward wearer's left
    row_b = (C['y_top'] - y) * K_BACK + 4
    back = sample(C['coat_back'], 190 - sb * K_BACK, row_b)
    w = smoothstep(np.pi / 2 - 0.06, np.pi / 2 + 0.06, np.abs(t))
    col = mix(front[:, :3], back[:, :3], w)
    # faint side seam line
    seam = np.exp(-((np.abs(t) - np.pi / 2) * np.sqrt(a * b) / 0.0015) ** 2)
    col = col * (1 - 0.10 * seam[:, None])
    return col


def coat_plain(Pt, scale=900.0):
    """Plain fabric (mirror-tiled swatch) for collar, sleeves inner etc."""
    sw = _coat.get('swatch')
    if sw is None:
        sw = flatten_lowfreq(piece('coat_back')[200:300, 70:170], 10)
        _coat['swatch'] = sw
    H, W = sw.shape[:2]
    u = (Pt[:, 0] + Pt[:, 2] * 0.7) * scale
    v = (Pt[:, 1] + Pt[:, 2] * 0.3) * scale
    def mt(c, n):
        k = np.floor(c / n); f = c - k * n
        return np.where(k % 2 == 0, f, n - 1 - f)
    return sample(sw, mt(u, W - 2) + 1, mt(v, H - 2) + 1)[:, :3]


def paint_sleeve(Pt, N, side_sign):
    q = Pt.copy(); q[:, 0] = np.abs(q[:, 0])
    name = 'sleeve_a' if side_sign < 0 else 'sleeve_b'
    x0, y0, x1, y1 = META[name]['bbox']
    phi = np.arctan2(q[:, 2] + 0.067, q[:, 1] - 1.443)      # 0 = top, + = front
    v = (q[:, 0] - 0.17) / (P.SLEEVE_END - 0.17)
    px = (x0 + x1) / 2 + np.sign(side_sign) * (phi / np.pi) * (x1 - x0) / 2 * 0.96
    py = y0 + 30 + v * (y1 - y0 - 34)
    return sample(piece(name), px, py)[:, :3]


def paint_coat(Pt, N):
    body = coat_uv(Pt)
    ax = np.abs(Pt[:, 0])
    # sleeves start at the armhole
    arm_line = 0.192 + 0.012 * np.clip((1.47 - Pt[:, 1]) / 0.1, -1, 1)
    w_s = smoothstep(arm_line - 0.004, arm_line + 0.004, ax) * (Pt[:, 1] > 1.30)
    sl = np.where((Pt[:, 0] > 0)[:, None], paint_sleeve(Pt, N, 1), paint_sleeve(Pt, N, -1))
    col = mix(body, sl, w_s)
    seam = np.exp(-((ax - arm_line) / 0.0018) ** 2) * (Pt[:, 1] > 1.32)
    return col * (1 - 0.12 * seam[:, None])


def paint_cuff(Pt, N, side='L'):
    q = Pt.copy(); q[:, 0] = np.abs(q[:, 0])
    name = 'cuff_a' if side == 'R' else 'cuff_b'
    x0, y0, x1, y1 = META[name]['bbox']
    phi = np.arctan2(q[:, 2] + 0.069, q[:, 1] - 1.441)
    u = (phi / (2 * np.pi)) % 1.0
    u = np.where(u > 0.5, 1 - u, u) * 2
    h = q[:, 0] - (P.SLEEVE_END - 0.004)
    v = np.clip((h + 0.034) / 0.068, 0, 1)
    return sample(piece(name), x0 + 3 + u * (x1 - x0 - 6), y1 - 3 - v * (y1 - y0 - 6))[:, :3]


def paint_collar(Pt, N):
    col = coat_plain(Pt, 1100.0)
    # edge stitching line along the outer fall edge
    rn = np.sqrt(Pt[:, 0] ** 2 + (Pt[:, 2] + 0.016) ** 2)
    st = np.exp(-((Pt[:, 1] - 1.468 - (rn - 0.13) * -0.6) / 0.0012) ** 2) * (rn > 0.11)
    return col * (1 - 0.10 * st[:, None])
