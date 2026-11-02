"""Shape definitions for every hero part, as signed distance fields (or an
explicit grid for the simulated coat skirt). glTF axes, metres."""
import os, sys, json
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from sdf import *

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))

# ------------------------------------------------------------------ skeleton
_man = np.load(os.path.join(ROOT, 'build/mannequin.npz'))
BONES = list(_man['bones'])
HEAD = {b: _man['heads'][i] for i, b in enumerate(BONES)}
TAIL = {b: _man['tails'][i] for i, b in enumerate(BONES)}

# ------------------------------------------------------------------ body sdf
_b = np.load(os.path.join(ROOT, 'build/body_sdf.npz'))
BODY_GRID = Grid(_b['lo'], _b['lo'] + _b['h'] * (np.array(_b['shape']) - 1), float(_b['h']))
BODY_SDF = _b['sdf']


def body(p):
    return BODY_GRID.sample(BODY_SDF, p, order=1, cval=0.5)


def mirror_x(fn):
    """Evaluate a +X-side shape on both sides."""
    def f(p):
        q = p.copy(); q[:, 0] = np.abs(q[:, 0])
        return fn(q)
    return f

# ------------------------------------------------------------------ face landmarks (match the face photo)
Y_EYE = 1.699
EYE_X = 0.0315
Y_CHIN = 1.583
Y_MOUTH = 1.632
Y_NOSE = 1.661
Y_BROW = 1.713
Y_HAIRLINE = 1.751
SCALP_TOP = 1.815


def _g(x, y, cx, cy, sx, sy):
    return np.exp(-(((x - cx) / sx) ** 2 + ((y - cy) / sy) ** 2))


def face_displacement(p):
    """Outward relief (metres) layered on the smooth base head; features are
    placed on the landmarks of the face photo so geometry and texture agree."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    ax = np.abs(x)
    D = np.zeros(len(p))
    # nose: bridge ramps up into the tip
    ramp = np.clip((1.707 - y) / (1.707 - 1.668), 0, 1)
    bridge = (0.003 + 0.0165 * ramp ** 1.5) * np.exp(-(x / (0.0062 + 0.0045 * ramp)) ** 2)
    bridge *= np.clip((y - 1.6585) / 0.006, 0, 1) * np.clip((1.716 - y) / 0.008, 0, 1)
    D += bridge
    D += 0.0060 * _g(x, y, 0, 1.6665, 0.0090, 0.0065)                       # tip
    D += 0.0068 * (_g(ax, y, 0.0128, 1.6630, 0.0058, 0.0052))                # alae
    D += 0.0040 * _g(x, y, 0, 1.6600, 0.0060, 0.0030)                        # columella
    # brow, sockets, lids, cheekbones
    D += 0.0042 * _g(x, y, 0, 1.7135, 0.036, 0.0055)
    D += 0.0025 * _g(ax, y, 0.030, 1.7150, 0.016, 0.0045)
    D -= 0.0085 * _g(ax, y, EYE_X, 1.7000, 0.0135, 0.0080)
    D += 0.0052 * _g(ax, y, EYE_X, Y_EYE, 0.0098, 0.0060)                    # eyeball / lids
    D -= 0.0018 * _g(ax, y, EYE_X + 0.002, 1.6905, 0.010, 0.0032)          # lower-lid crease
    D += 0.0032 * _g(ax, y, 0.045, 1.6815, 0.013, 0.0085)                    # cheekbones
    D -= 0.0020 * _g(ax, y, 0.022, 1.6560, 0.006, 0.012)                     # nasolabial
    # mouth
    D += 0.0030 * _g(x, y, 0, 1.6460, 0.013, 0.0070)                         # philtrum bulge
    D += 0.0040 * _g(x, y, 0, 1.6362, 0.0165, 0.0036)                        # upper lip
    D += 0.0042 * _g(x, y, 0, 1.6268, 0.0145, 0.0042)                        # lower lip
    D -= 0.0030 * _g(x, y, 0, Y_MOUTH, 0.0185, 0.0011)                       # lip seam
    D -= 0.0018 * _g(ax, y, 0.0245, Y_MOUTH, 0.004, 0.004)                   # mouth corners
    D -= 0.0028 * _g(x, y, 0, 1.6165, 0.012, 0.0035)                         # mentolabial fold
    D += 0.0040 * _g(x, y, 0, 1.5985, 0.0175, 0.0085)                        # chin
    w = np.clip((z - 0.035) / 0.035, 0, 1)
    return D * w


def head_base(p):
    d = ellipsoid(p, (0, 1.716, -0.012), (0.0735, 0.0985, 0.0975))           # cranium
    d = smin(d, ellipsoid(p, (0, 1.662, 0.020), (0.0605, 0.0760, 0.0770)), 0.022)   # face mass
    d = smin(d, ellipsoid(p, (0, 1.636, 0.044), (0.038, 0.032, 0.0505)), 0.020)     # maxilla / muzzle
    jaw = capsule(np.c_[np.abs(p[:, 0]), p[:, 1:]], (0.052, 1.618, -0.012), (0.018, 1.592, 0.066), 0.012, 0.015)
    d = smin(d, jaw, 0.022)
    d = smin(d, ellipsoid(p, (0, 1.597, 0.070), (0.021, 0.015, 0.0195)), 0.014)      # chin mass
    neck = capsule(p, (0, 1.395, -0.022), (0, 1.630, -0.020), 0.0590, 0.0550)
    neck = smin(neck, ellipsoid(p, (0, 1.548, 0.039), (0.0095, 0.013, 0.010)), 0.010)
    d = smin(d, neck, 0.030)
    return d


def head_shape(p):
    """Head + neck sculpt (first version, chosen by the user): explicit
    eye sockets/eyeballs, lips, cheekbones, brow ridge, placed on the face
    photo's landmarks."""
    d = ellipsoid(p, (0, 1.718, -0.010), (0.0745, 0.097, 0.097))               # cranium
    d = smin(d, ellipsoid(p, (0, 1.735, 0.030), (0.063, 0.062, 0.064)), 0.02)  # forehead
    d = smin(d, ellipsoid(p, (0, 1.664, 0.026), (0.0655, 0.062, 0.066)), 0.02) # midface / cheeks
    d = smin(d, ellipsoid(p, (0, 1.622, 0.020), (0.059, 0.046, 0.062)), 0.02)  # lower face
    # mandible: jaw angles to chin
    jaw = capsule(np.c_[np.abs(p[:, 0]), p[:, 1:]], (0.057, 1.617, -0.010), (0.020, 1.590, 0.070), 0.0145, 0.016)
    d = smin(d, jaw, 0.018)
    d = smin(d, ellipsoid(p, (0, 1.595, 0.073), (0.025, 0.016, 0.018)), 0.012)  # chin
    # cheekbones and brow ridge
    d = smin(d, mirror_x(lambda q: ellipsoid(q, (0.047, 1.684, 0.064), (0.022, 0.012, 0.020)))(p), 0.012)
    d = smin(d, capsule(p, (-0.047, 1.7135, 0.079), (0.047, 1.7135, 0.079), 0.0105), 0.016)
    # eye sockets, then eyeballs/lids
    d = smax(d, -mirror_x(lambda q: ellipsoid(q, (EYE_X, 1.7005, 0.095), (0.0175, 0.0115, 0.0125)))(p), 0.008)
    d = smin(d, mirror_x(lambda q: sphere(q, (EYE_X, Y_EYE, 0.0775), 0.0128))(p), 0.0045)
    # nose
    d = smin(d, capsule(p, (0, 1.707, 0.087), (0, 1.671, 0.1105), 0.0068, 0.0092), 0.010)
    d = smin(d, sphere(p, (0, 1.6665, 0.1085), 0.0112), 0.008)
    d = smin(d, mirror_x(lambda q: sphere(q, (0.0135, 1.6635, 0.0975), 0.0083))(p), 0.007)
    d = smax(d, -mirror_x(lambda q: ellipsoid(q, (0.0072, 1.6575, 0.1035), (0.0042, 0.0026, 0.0052)))(p), 0.002)
    # mouth
    d = smin(d, ellipsoid(p, (0, 1.6415, 0.0925), (0.026, 0.0105, 0.0105)), 0.008)   # upper lip / philtrum
    d = smin(d, ellipsoid(p, (0, 1.6355, 0.0975), (0.0225, 0.0055, 0.0080)), 0.004)  # upper lip
    d = smin(d, ellipsoid(p, (0, 1.6268, 0.0965), (0.0200, 0.0066, 0.0082)), 0.004)  # lower lip
    d = smax(d, -ellipsoid(p, (0, Y_MOUTH, 0.1035), (0.0235, 0.0011, 0.0065)), 0.0015)  # lip seam
    d = smax(d, -ellipsoid(p, (0, 1.6165, 0.0955), (0.016, 0.0038, 0.0055)), 0.006)  # mentolabial fold
    # neck (+ adam's apple)
    neck = capsule(p, (0, 1.395, -0.022), (0, 1.630, -0.018), 0.0585, 0.0545)
    neck = smin(neck, ellipsoid(p, (0, 1.548, 0.040), (0.010, 0.014, 0.010)), 0.01)
    d = smin(d, neck, 0.028)
    return d


def head_sdf(p):
    # bottom cut sits inside the tee collar
    return smax(head_shape(p), 1.405 - p[:, 1], 0.004)


_MH = {}


def mh_head_sdf(p):
    """Signed distance to the MakeHuman head surface (nearest dense sample,
    sign from its normal) - accurate near the scalp, where the hair lives."""
    if 'tree' not in _MH:
        from scipy.spatial import cKDTree
        d = np.load(os.path.join(ROOT, 'build/cache/mh_head.npz'))
        V, F = d['V'], d['F']
        tri = V[F]
        fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        VN = np.zeros_like(V)
        for k in range(3):
            np.add.at(VN, F[:, k], fn)
        VN /= np.maximum(np.linalg.norm(VN, axis=1, keepdims=True), 1e-12)
        # dense samples on each triangle (barycentric grid)
        bs = np.array([(a, b, 1 - a - b) for a in np.linspace(0, 1, 5) for b in np.linspace(0, 1, 5) if a + b <= 1.0001])
        S = np.einsum('kj,tjc->tkc', bs, tri).reshape(-1, 3)
        SN = np.einsum('kj,tjc->tkc', bs, VN[F]).reshape(-1, 3)
        SN /= np.maximum(np.linalg.norm(SN, axis=1, keepdims=True), 1e-12)
        _MH.update(tree=cKDTree(S), S=S, N=SN)
    dist, idx = _MH['tree'].query(p)
    sgn = np.sign(((p - _MH['S'][idx]) * _MH['N'][idx]).sum(1))
    return dist * np.where(sgn == 0, 1, sgn)


HEAD_BOX = ((-0.095, 1.395, -0.125), (0.095, 1.83, 0.135))


HEAD_CZ = 0.022       # front/back centre of the (MakeHuman) cranium


def hair_line(p):
    """Border between the long slicked-back top and the clipper-short sides
    (client's head video): natural forehead hairline, a high soft border along
    the parietal ridge, coming down to the bun at the back."""
    th = np.degrees(np.abs(np.arctan2(p[:, 0], p[:, 2] - HEAD_CZ)))       # 0 = front
    _tw = json.load(open(os.path.join(os.path.dirname(__file__), 'face_tweaks.json'))).get('hairline')
    if _tw:
        return np.interp(th, _tw[0], _tw[1])
    return np.interp(th, [0, 14, 28, 42, 60, 85, 110, 135, 158, 180],
                     [1.7480, 1.7510, 1.7580, 1.7660, 1.7720, 1.7750, 1.7720, 1.7640, 1.7520, 1.7450])


def hair_sdf(p):
    """Slicked-back top section of the undercut: a shell over the scalp,
    thickest on top, swept toward the bun, with a crisp edge at the
    undercut line."""
    base = mh_head_sdf(p)
    above = p[:, 1] - hair_line(p)
    front = np.clip((p[:, 2] - HEAD_CZ) / 0.09, 0, 1)            # 1 at the forehead
    # combed tight to the skull: ~2-3 mm, a slightly raised soft edge
    s = np.clip(above / 0.02, 0, 1)
    edge = 0.0014 + 0.0004 * (1 - front)
    thick = edge + 0.0013 * s + 0.0009 * front * s
    d = base - thick
    d = smax(d, -above, 0.0025)                       # undercut line
    d = smax(d, -(base + 0.0006), 0.001)              # never below the scalp
    return d


HAIR_BOX = ((-0.09, 1.725, -0.12), (0.09, 1.84, 0.165))

# ------------------------------------------------------------------ ears
EAR_CENTER = np.array([0.0745, 1.683, -0.012])


def _ear_frame():
    # ear plane faces outward (+x) with a slight backward/upward tilt
    n = np.array([0.93, 0.0, 0.37]); n /= np.linalg.norm(n)       # outward normal (points a bit forward)
    up = np.array([0.0, 1.0, 0.0]); up -= n * (up @ n); up /= np.linalg.norm(up)
    back = np.cross(up, n)                                          # roughly -z
    # tilt the ear's long axis back by 12 degrees
    a = np.radians(12)
    up2 = np.cos(a) * up - np.sin(a) * back * -1
    back2 = np.cross(up2, n)
    return n, up2 / np.linalg.norm(up2), back2 / np.linalg.norm(back2)


EAR_N, EAR_UP, EAR_BACK = _ear_frame()


def ear_sdf_left(p):
    q = p - EAR_CENTER
    lx, ly, lz = q @ EAR_N, q @ EAR_UP, q @ EAR_BACK      # out, up, back
    L = np.c_[lz, ly, lx]
    # ear disc: ellipse 0.03 x 0.0215 (half), thin, cupped
    shell = ellipsoid(L, (0.004, 0.0, 0.004), (0.0215, 0.031, 0.0085))
    # hollow out the concha on the outer face
    shell = smax(shell, -ellipsoid(L, (0.000, -0.004, 0.0105), (0.0125, 0.016, 0.007)), 0.003)
    # attach to the skull
    shell = smin(shell, ellipsoid(L, (-0.006, -0.004, -0.007), (0.012, 0.020, 0.008)), 0.006)
    return shell


EAR_BOX = ((0.040, 1.635, -0.055), (0.105, 1.735, 0.030))

# ------------------------------------------------------------------ hair bun (on the crown, slightly back)
BUN_AXIS = np.array([0.0, 0.18, -0.98]); BUN_AXIS /= np.linalg.norm(BUN_AXIS)   # low on the back of the head (client video)
BUN_CENTER = np.array([0.0, 1.808, -0.083])


def place_bun():
    """Seat the bun on the MakeHuman scalp at the crown, behind the top."""
    global BUN_CENTER
    c = np.array([0.0, 1.712, HEAD_CZ])
    ts = np.linspace(0.02, 0.2, 600)
    pts = c + ts[:, None] * BUN_AXIS
    k = int(np.argmax(mh_head_sdf(pts) > 0))
    BUN_CENTER = pts[k] + BUN_AXIS * 0.016


def bun_sdf(p):
    if not _MH.get('bun'):
        place_bun(); _MH['bun'] = True
    q = p - BUN_CENTER
    h = q @ BUN_AXIS
    radial = q - h[:, None] * BUN_AXIS
    rr = length(radial)
    # twisted torus-ish knot: radius modulated by angle for coil bumps
    ref = np.cross(BUN_AXIS, [1.0, 0, 0]); ref /= np.linalg.norm(ref)
    ang = np.arctan2(radial @ np.cross(BUN_AXIS, ref), radial @ ref)
    coil = 0.0022 * np.sin(5 * ang + 9 * h / 0.03)
    body_ = ellipsoid(np.c_[rr, h, np.zeros_like(h)], (0, 0.001, 0), (0.021, 0.015, 0.021)) + coil * 0.6
    tie = torus(p, BUN_CENTER - BUN_AXIS * 0.011, BUN_AXIS, 0.0125, 0.0024)
    return smin(body_, tie, 0.002)


BUN_BOX = ((-0.055, 1.72, -0.15), (0.055, 1.88, 0.03))


def tie_sdf(p):
    if not _MH.get('bun'):
        place_bun(); _MH['bun'] = True
    return torus(p, BUN_CENTER - BUN_AXIS * 0.011, BUN_AXIS, 0.0125, 0.0024)

# ------------------------------------------------------------------ t-shirt
TEE_HEM = 0.958


def tee_sdf(p):
    d = body(p) - 0.0105
    # straight-hanging lower body (tees don't hug the waist)
    drape = ellipsoid(p, (0, 1.08, 0.008), (0.150, 0.30, 0.118))
    d = smin(d, smax(drape, p[:, 1] - 1.30, 0.04), 0.03)
    d = smax(d, TEE_HEM - p[:, 1], 0.003)                       # hem
    d = smax(d, np.abs(p[:, 0]) - 0.285, 0.004)                 # short sleeves (hidden under the coat)
    ax = np.array([0.0, 1.0, -0.33]); ax /= np.linalg.norm(ax)
    d = smax(d, (p - np.array([0.0, 1.488, -0.008])) @ ax, 0.004)            # crew neckline
    # snug round neck hole matching the rib (no gap around the neck)
    q = p - np.array([0.0, 1.488, -0.008])
    h = q @ ax
    rn = np.linalg.norm(q - h[:, None] * ax, axis=1)
    d = smax(d, (0.0645 - rn) - np.maximum(-h - 0.03, 0) * 2, 0.003)
    return d


TEE_BOX = ((-0.30, 0.94, -0.20), (0.30, 1.56, 0.19))


def collar_rib_sdf(p):
    c = np.array([0.0, 1.488, -0.008])
    ax = np.array([0.0, 1.0, -0.33]); ax /= np.linalg.norm(ax)
    return torus(p, c - ax * 0.002, ax, 0.0660, 0.0058)

# ------------------------------------------------------------------ jeans
JEANS_TOP = 1.065
JEANS_HEM = 0.088


def jeans_sdf(p):
    d = body(p) - 0.012
    for s in (-1, 1):
        x = 0.089 * s
        leg = capsule(p, (x * 1.02, 0.86, 0.004), (x * 1.06, 0.53, 0.008), 0.073, 0.060)
        leg = smin(leg, capsule(p, (x * 1.06, 0.53, 0.008), (x * 1.06, JEANS_HEM - 0.02, -0.022), 0.060, 0.056), 0.02)
        d = smin(d, leg, 0.02)
    d = smax(d, p[:, 1] - JEANS_TOP, 0.003)
    d = smax(d, JEANS_HEM - p[:, 1], 0.003)
    # drop arms/torso above the waist (just in case)
    d = smax(d, np.abs(p[:, 0]) - 0.20, 0.01)
    return d


JEANS_BOX = ((-0.21, 0.07, -0.17), (0.21, 1.08, 0.17))

# ------------------------------------------------------------------ shoes (left; mirrored for right)
def shoe_sdf_left(p):
    x0 = 0.089
    q = p.copy()
    # sole: rounded slab with toe spring
    toe_lift = np.clip((q[:, 2] - 0.12) / 0.09, 0, 1) ** 2 * 0.018
    qs = q.copy(); qs[:, 1] -= toe_lift
    width = 0.047 + 0.006 * np.clip((q[:, 2] - 0.0) / 0.12, -1, 1)   # wider at forefoot
    sole = box(np.c_[(qs[:, 0] - x0) / width * 0.05, qs[:, 1], qs[:, 2]], (0, 0.016, 0.060), (0.05, 0.016, 0.148), 0.012)
    # upper
    up = ellipsoid(q, (x0, 0.062, -0.040), (0.046, 0.058, 0.058))
    up = smin(up, ellipsoid(q, (x0 + 0.002, 0.055, 0.040), (0.049, 0.050, 0.085)), 0.03)
    up = smin(up, ellipsoid(q, (x0 + 0.004, 0.040 + 0.008, 0.140), (0.046, 0.028, 0.070)), 0.03)
    up = smax(up, 0.018 - q[:, 1], 0.004)
    d = smin(sole, up, 0.006)
    # collar opening around the ankle
    d = smax(d, -capsule(q, (x0, 0.11, -0.040), (x0, 0.25, -0.05), 0.040), 0.008)
    d = smax(d, q[:, 1] - 0.122, 0.006)
    return d


SHOE_BOX = ((0.02, -0.01, -0.12), (0.16, 0.14, 0.24))

# ------------------------------------------------------------------ forearm + hand (left; mirrored)
def finger_chain(side, name):
    pts = [HEAD[f'{name}_01_{side}'], HEAD[f'{name}_02_{side}'], HEAD[f'{name}_03_{side}'], TAIL[f'{name}_03_{side}']]
    tip = pts[-1] + (pts[-1] - pts[-2]) * 0.55
    return pts[:-1] + [tip]


FINGER_R = {'index': (0.0098, 0.0088, 0.0080, 0.0072), 'middle': (0.0102, 0.0091, 0.0082, 0.0074),
            'ring': (0.0096, 0.0086, 0.0078, 0.0070), 'pinky': (0.0084, 0.0076, 0.0069, 0.0062),
            'thumb': (0.0135, 0.0112, 0.0098, 0.0086)}


def hand_parts_left():
    """List of (bone_name, sdf_fn) primitives making up the left forearm+hand."""
    parts = []
    el = HEAD['lowerarm_l']; wr = HEAD['hand_l']
    ax = (wr - el) / np.linalg.norm(wr - el)

    def forearm(p):
        q = p - el
        h = np.clip(q @ ax / np.linalg.norm(wr - el), 0, 1)
        c = el + h[:, None] * (wr - el)
        r = q - (h[:, None] * (wr - el))
        # flattened cross-section towards the wrist (wider in z than y)
        sy = 1.0 + 0.30 * h
        rr = np.sqrt(r[:, 0] ** 2 + (r[:, 1] * sy) ** 2 + r[:, 2] ** 2)
        rad = 0.041 - 0.0135 * h + 0.004 * np.exp(-((h - 0.22) / 0.18) ** 2)
        d = rr - rad
        d = np.where((q @ ax) < -0.02, np.maximum(d, -0.02 - q @ ax), d)
        return d
    parts.append(('lowerarm_l', forearm))
    # palm
    pc = (HEAD['hand_l'] + HEAD['middle_01_l']) * 0.5 + np.array([0.004, -0.002, -0.006])
    parts.append(('hand_l', lambda p: box(p, pc, (0.054, 0.0145, 0.042), 0.012)))
    parts.append(('hand_l', lambda p: ellipsoid(p, HEAD['hand_l'] + np.array([0.022, -0.003, -0.012]), (0.03, 0.019, 0.034))))
    # thenar eminence
    parts.append(('thumb_01_l', lambda p: ellipsoid(p, HEAD['thumb_01_l'] + np.array([0.012, -0.006, -0.012]), (0.026, 0.014, 0.018))))
    for name in ['index', 'middle', 'ring', 'pinky', 'thumb']:
        pts = finger_chain('l', name)
        rs = FINGER_R[name]
        for k in range(3):
            a, b, ra, rb = pts[k], pts[k + 1], rs[k], rs[k + 1]
            parts.append((f'{name}_0{k+1}_l', (lambda a, b, ra, rb: lambda p: capsule(p, a, b, ra, rb))(a, b, ra, rb)))
    return parts


HAND_PARTS = hand_parts_left()


def hand_sdf_left(p):
    d = None
    for bone, fn in HAND_PARTS:
        v = fn(p)
        if d is None:
            d = v
        elif bone in ('hand_l',):
            d = smin(d, v, 0.016)
        elif bone == 'thumb_01_l':
            d = smin(d, v, 0.009)
        elif bone.endswith('_01_l'):
            d = smin(d, v, 0.006)
        else:
            d = smin(d, v, 0.0035)
    # sleeve end is inside the rolled cuff
    d = smax(d, HEAD['lowerarm_l'][0] - 0.035 - p[:, 0], 0.003)
    return d


HAND_BOX = ((0.42, 1.33, -0.14), (1.00, 1.49, 0.05))

# ------------------------------------------------------------------ coat (skinned upper part)
COAT_SPLIT = 1.022          # (legacy) waist height used for texture layout references
COAT_HEM = 0.505
COAT_T = 0.0065             # cloth thickness
SLEEVE_END = 0.425          # x where the sleeve is rolled up (just above the elbow at 0.466)


def coat_outer_torso(p):
    """Coat body without sleeves (arms clipped at the armholes)."""
    d = body(p) - 0.026
    d = smax(d, np.abs(p[:, 0]) - 0.215, 0.02)
    drape = ellipsoid(p, (0, 1.12, 0.004), (0.178, 0.36, 0.142))
    return smin(d, smax(drape, p[:, 1] - 1.36, 0.05), 0.04)


def coat_outer(p):
    d = body(p) - 0.026
    drape = ellipsoid(p, (0, 1.12, 0.004), (0.178, 0.36, 0.142))
    d = smin(d, smax(drape, p[:, 1] - 1.36, 0.05), 0.04)
    # loose sleeves
    for s in (-1, 1):
        sh = np.array([0.175 * s, 1.445, -0.064])
        el = np.array([(SLEEVE_END + 0.04) * s, 1.441, -0.070])
        d = smin(d, capsule(p, sh, el, 0.066, 0.058), 0.035)
    return d


def ref_ellipse(y):
    """Reference cross-section (centre z, half-width a, half-depth b) used to
    unwrap coat/skirt coordinates. Smooth in y."""
    y = np.asarray(y, float)
    zc = 0.004 + 0.0 * y
    a = np.interp(y, [0.4, 0.95, 1.05, 1.25, 1.40, 1.6], [0.215, 0.205, 0.184, 0.180, 0.185, 0.185])
    b = np.interp(y, [0.4, 0.95, 1.05, 1.25, 1.40, 1.6], [0.165, 0.150, 0.140, 0.146, 0.150, 0.150])
    return zc, a, b


def ellipse_angle(p):
    zc, a, b = ref_ellipse(p[:, 1])
    return np.arctan2(p[:, 0] / a, (p[:, 2] - zc) / b)   # 0 = front centre, +pi/2 = wearer's left side


def coat_open_angle(y):
    """Half-angle of the open front (parametric ellipse angle) as a function of height."""
    return np.radians(np.interp(y, [0.4, 1.0, 1.17, 1.25, 1.33, 1.40, 1.46, 1.50, 1.60],
                                   [22.0, 22.0, 23.0, 26.5, 34.0, 43.0, 54.0, 64.0, 70.0]))


def yoke_line(t):
    """Height of the seam between the rigid shoulder yoke and the free-hanging
    coat body, as a function of the parametric angle around the torso.
    Front: just under the collarbone; sides: the armpits (attached to the
    sleeves); back: across the shoulder blades."""
    a = np.degrees(np.abs(t))
    return np.interp(a, [0, 50, 75, 90, 105, 130, 180], [1.445, 1.438, 1.398, 1.382, 1.378, 1.372, 1.375])


def coat_sdf(p):
    o = coat_outer(p)
    shell = np.abs(o + COAT_T * 0.5) - COAT_T * 0.5
    t = np.abs(ellipse_angle(p))
    zc, a, b = ref_ellipse(p[:, 1])
    rr = np.sqrt((p[:, 0]) ** 2 + (p[:, 2] - zc) ** 2)
    opening = (coat_open_angle(p[:, 1]) - t) * rr
    torso_zone = np.abs(p[:, 0]) < 0.24
    shell = np.where(torso_zone, smax(shell, opening, 0.0015), shell)
    # neck opening
    rn = length(np.c_[p[:, 0], p[:, 2] + 0.016])
    neck_cut = (0.080 - rn) - np.maximum(1.455 - p[:, 1], 0) * 3
    shell = smax(shell, neck_cut, 0.002)
    # everything below the yoke seam is simulated cloth (not part of this rigid mesh)
    yoke_cut = yoke_line(ellipse_angle(p)) - p[:, 1]
    yoke_cut = np.where(np.abs(p[:, 0]) < 0.205, yoke_cut, np.maximum(yoke_cut, 1.355 - p[:, 1]))
    shell = smax(shell, yoke_cut, 0.0015)
    shell = smax(shell, np.abs(p[:, 0]) - SLEEVE_END, 0.0015)
    return shell


COAT_BOX = ((-0.47, 1.33, -0.23), (0.47, 1.60, 0.22))


def seg_dist2d(px, py, pts):
    """Distance from 2D points to a polyline."""
    best = np.full(px.shape, np.inf)
    for (ax_, ay), (bx, by) in zip(pts[:-1], pts[1:]):
        vx, vy = bx - ax_, by - ay
        t = np.clip(((px - ax_) * vx + (py - ay) * vy) / (vx * vx + vy * vy), 0, 1)
        dx, dy = px - ax_ - t * vx, py - ay - t * vy
        best = np.minimum(best, np.sqrt(dx * dx + dy * dy))
    return best


COLLAR_PROFILE = [(0.080, 1.460), (0.0788, 1.476), (0.0812, 1.4865), (0.088, 1.4885), (0.100, 1.4825),
                  (0.118, 1.4705), (0.139, 1.4565)]


def coat_collar_sdf(p):
    """Fold-down shirt-style collar: in every angular slice around the neck
    the cross-section is a folded strip (inner stand + outer fall)."""
    c = np.array([0.0, 0.0, -0.016])
    rx, rz = p[:, 0] - c[0], p[:, 2] - c[2]
    rn = np.sqrt(rx * rx + rz * rz)
    ang = np.abs(np.arctan2(rx, rz))                                    # 0 = front
    # the fall widens toward the front points
    front = np.clip((np.radians(105) - ang) / np.radians(50), 0, 1)
    prof = [(r + 0.006 * front * (k >= 4) * (k - 3), y - 0.010 * front * (k >= 5)) for k, (r, y) in enumerate(COLLAR_PROFILE)]
    d = None
    rn_adj = rn
    d = seg_dist2d(rn_adj, p[:, 1], np.array(COLLAR_PROFILE)) - 0.0028
    # angular extent: wraps the back and sides, ends at the lapel notch
    d = smax(d, (np.radians(52) - ang) * rn, 0.003)
    return d


def cuff_sdf_left(p):
    c = np.array([SLEEVE_END - 0.004, 1.441, -0.069])
    ax = np.array([1.0, 0, 0])
    q = p - c
    h = q[:, 0]
    rr = length(q[:, 1:])
    # rolled band: rounded rectangle cross-section (width along arm 0.062, thickness 0.016)
    cs = np.c_[h, rr - 0.0615]
    d = box(np.c_[cs, np.zeros_like(h)], (0.0, 0.0, 0.0), (0.031, 0.0085, 1.0), 0.0075)
    return d


CUFF_BOX = ((SLEEVE_END - 0.05, 1.36, -0.15), (SLEEVE_END + 0.045, 1.53, 0.01))
