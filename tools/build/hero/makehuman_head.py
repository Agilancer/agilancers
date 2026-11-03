"""Generate the hero's head with MakeHuman (CC0 base mesh + morph targets).

Downloads are cached in build/makehuman (base.obj, targets/*.target, eyes/*)
from github.com/makehumancommunity/makehuman (data released as CC0). We
apply MakeHuman's macro targets (young, muscular, Caucasian male) and a set
of face modifiers chosen to match the supplied face photo (square jaw,
prominent chin, high cheekbones, lean cheeks, straight nose, low brows),
cut out the head + neck (MakeHuman's own ears included), add MakeHuman's
high-poly eyeballs, and place it on the UAL2 rig's head."""
import os, sys, json, gzip
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
MH = os.path.join(ROOT, 'build', 'makehuman')
SRC = 'https://raw.githubusercontent.com/makehumancommunity/makehuman/master/makehuman/data/'

# modifier -> weight (file names follow data/targets/<group>/<name>.target)
MACRO = {
    # ~40 years old, solid build (client's own head is the reference)
    'macrodetails/universal-male-young-averagemuscle-averageweight': 0.45,
    'macrodetails/universal-male-young-maxmuscle-averageweight': 0.15,
    'macrodetails/universal-male-young-averagemuscle-maxweight': 0.20,
    'macrodetails/universal-male-old-averagemuscle-averageweight': 0.20,
    'macrodetails/caucasian-male-young': 0.8,
    'macrodetails/caucasian-male-old': 0.2,
}
FACE = {
    # rounder, fuller face with a softer jaw, broader nose, thick neck
    'head/head-round': 0.45,
    'head/head-square': 0.15,
    'head/head-fat-incr': 0.25,
    'chin/chin-width-incr': 0.25,
    'chin/chin-prominent-incr': 0.10,
    'cheek/l-cheek-volume-incr': 0.30, 'cheek/r-cheek-volume-incr': 0.30,
    'cheek/l-cheek-bones-incr': 0.10, 'cheek/r-cheek-bones-incr': 0.10,
    'nose/nose-scale-horiz-incr': 0.20,
    'nose/nose-point-width-incr': 0.20,
    'nose/nose-scale-vert-decr': 0.20,
    'mouth/mouth-scale-horiz-incr': 0.30,
    'eyebrows/eyebrows-trans-down': 0.15,
    'neck/neck-scale-horiz-incr': 0.50,
    'neck/neck-scale-depth-incr': 0.40,
    'neck/neck-double-incr': 0.15,
}


FIT_PATH = os.path.join(ROOT, 'build', 'cache', 'face_fit.json')
if os.path.exists(FIT_PATH):
    FACE = {k: v for k, v in json.load(open(FIT_PATH))['weights'].items() if abs(v) > 1e-3}
    FACE.update({'neck/neck-scale-horiz-incr': 0.50, 'neck/neck-scale-depth-incr': 0.40, 'neck/neck-double-incr': 0.15})
_tw = json.load(open(os.path.join(HERE, 'face_tweaks.json')))
for _k, _v in _tw.get('morph_add', {}).items():
    FACE[_k] = FACE.get(_k, 0.0) + _v


def fetch(rel, dst):
    if os.path.exists(dst):
        return dst
    import urllib.request
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with urllib.request.urlopen(SRC + rel, timeout=60) as r:
        open(dst, 'wb').write(r.read())
    return dst


def load_obj(path, groups_keep=None):
    V, VT, F, FT, G = [], [], [], [], []
    g = None
    for l in open(path):
        if l.startswith('v '):
            V.append([float(x) for x in l.split()[1:4]])
        elif l.startswith('vt '):
            VT.append([float(x) for x in l.split()[1:3]])
        elif l.startswith('g '):
            g = l.split()[1]
        elif l.startswith('f '):
            toks = [t.split('/') for t in l.split()[1:]]
            F.append([int(t[0]) - 1 for t in toks])
            FT.append([int(t[1]) - 1 if len(t) > 1 and t[1] else -1 for t in toks])
            G.append(g)
    return np.array(V), np.array(VT) if VT else None, F, FT, G


def load_target(rel, n):
    path = fetch(f'targets/{rel}.target', os.path.join(MH, 'targets', rel.replace('/', '_') + '.target'))
    D = np.zeros((n, 3))
    for l in open(path):
        if not l.strip() or l.startswith('#'):
            continue
        t = l.split()
        D[int(t[0])] = [float(t[1]), float(t[2]), float(t[3])]
    return D


def triangulate(F, FT=None):
    T, TT = [], []
    for k, f in enumerate(F):
        for i in range(1, len(f) - 1):
            T.append([f[0], f[i], f[i + 1]])
            if FT is not None:
                ft = FT[k]; TT.append([ft[0], ft[i], ft[i + 1]])
    return np.array(T), (np.array(TT) if FT is not None else None)


def fit_proxy(mhclo, V, n):
    """MakeHuman proxy fitting: each proxy vertex = barycentric blend of three
    (morphed) base-mesh vertices + an offset scaled by the body's local size."""
    scales = {}
    rows = []
    in_verts = False
    for l in open(mhclo):
        t = l.split()
        if not t or t[0].startswith('#'):
            continue
        if t[0] in ('x_scale', 'y_scale', 'z_scale'):
            scales[t[0][0]] = (int(t[1]), int(t[2]), float(t[3]))
        elif t[0] == 'verts':
            in_verts = True
        elif in_verts:
            if len(t) == 9:
                rows.append([float(x) for x in t])
            elif len(t) == 1:
                rows.append([float(t[0]), float(t[0]), float(t[0]), 1, 0, 0, 0, 0, 0])
            else:
                in_verts = False
    R = np.array(rows[:n])
    idx = R[:, :3].astype(int)
    w = R[:, 3:6]
    off = R[:, 6:9]
    sc = np.ones(3)
    for k, ax in enumerate('xyz'):
        if ax in scales:
            a, b, s0 = scales[ax]
            sc[k] = abs(V[a, k] - V[b, k]) / s0
    return (V[idx] * w[..., None]).sum(1) + off * sc


def group_center(V, F, G, name):
    idx = sorted({i for f, g in zip(F, G) if g == name for i in f})
    return V[idx].mean(0)


def _ss(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def warp(Y, earw=None, nosew=None):
    """Direct proportion fixes the MakeHuman targets can't reach (measured on the
    client's frontal video frame, matched camera): a broader cranium that stays
    wide above the brows instead of tapering to an egg, a fuller jaw/jowl line,
    and a longer chin. Smooth, so the topology and the morphed features survive."""
    import parts as P
    w = json.load(open(os.path.join(HERE, 'face_tweaks.json'))).get('warp', {})
    Y = Y.copy()
    ye = P.Y_EYE
    rel = Y[:, 1] - ye
    ax = np.abs(Y[:, 0])
    # cranium: x grows from the eye line up (only the temples/ears near eye level, all of it from the forehead up)
    kc = w.get('cranium_x', 0.0) * _ss(-0.005, 0.07, rel)
    side_c = _ss(0.045, 0.07, ax) * (1 - _ss(0.025, 0.055, rel)) + _ss(0.025, 0.055, rel)
    # jaw/jowls: x grows below the cheekbones, fading out on the neck; the mouth/chin centre stays put
    kj = w.get('jaw_x', 0.0) * _ss(-0.005, -0.04, rel) * _ss(-0.125, -0.095, rel) * _ss(0.025, 0.060, ax)
    if earw is not None:
        kj = kj * (1 - earw)                         # the ears keep their own shape/position
    kb = 0.0
    if w.get('bridge_narrow', 0.0):
        # narrower nasal bridge (client: ~10 mm across vs ~17): pull the dorsum's sides toward the midline
        # between the nasion and the lobule; the tip, alae and nostrils keep their width
        kb = -w['bridge_narrow'] * _ss(0.019, 0.008, ax) * _ss(0.016, 0.006, rel) * _ss(-0.036, -0.024, rel)
    Y[:, 0] *= 1 + kc * side_c + kj + kb
    front = _ss(-0.01, 0.04, Y[:, 2] - P.HEAD_CZ)
    # forehead: more upright (client profile), fading out toward the temples and into the scalp
    ff = w.get('forehead_forward', 0.0)
    if ff:
        Y[:, 2] += ff * _ss(0.012, 0.065, rel) * _ss(0.12, 0.09, rel) * _ss(0.065, 0.030, ax) * front
    # underbite: lower lip, lip-chin fold and chin forward (below the mouth line), fading into the jaw and neck
    mf = w.get('mandible_forward', 0.0)
    if mf:
        Y[:, 2] += mf * _ss(-0.064, -0.071, rel) * _ss(-0.135, -0.105, rel) * _ss(0.050, 0.025, ax) * front
    ub = w.get('upper_lip_back', 0.0)
    if ub:
        Y[:, 2] -= ub * _ss(-0.046, -0.054, rel) * _ss(-0.071, -0.064, rel) * _ss(0.040, 0.022, ax) * front
    # crown: a taller dome (client's skull is higher above the hairline)
    Y[:, 1] += w.get('dome', 0.0) * _ss(0.065, 0.105, rel)
    # chin: stretch the face below the mouth line downward (front part only), fading out at the neck
    cd = w.get('chin_drop', 0.0)
    if cd:
        front = _ss(-0.03, 0.05, Y[:, 2] - P.HEAD_CZ)
        k = _ss(-0.055, -0.115, rel) * _ss(-0.165, -0.125, rel)
        Y[:, 1] -= cd * k * front
    return Y


def pack_uv(VT, TT, V3, T, margin=0.01):
    """MakeHuman's own UVs for the head (one continuous face island instead of
    hundreds of auto-unwrap islands), re-packed to fill the texture: islands keep
    their relative texel density and are shelf-packed into [0,1]^2.
    Returns per-corner UVs (n_tri, 3, 2)."""
    n = len(VT)
    parent = np.arange(n)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    for t in TT:
        r0 = find(t[0])
        for a in t[1:]:
            parent[find(a)] = r0
    roots = np.array([find(t[0]) for t in TT])
    isl = {r: i for i, r in enumerate(np.unique(roots))}
    tid = np.array([isl[r] for r in roots])
    UV = VT[TT].astype(np.float64)
    boxes = []
    for i in range(len(isl)):
        q = UV[tid == i].reshape(-1, 2)
        boxes.append((q.min(0), q.max(0)))
    def place(scale):
        order = sorted(range(len(boxes)), key=lambda i: -(boxes[i][1][1] - boxes[i][0][1]))
        x = y = shelf = margin; pos = {}
        for i in order:
            w, h = (boxes[i][1] - boxes[i][0]) * scale
            if x + w + margin > 1:
                x = margin; y += shelf + margin; shelf = 0
            pos[i] = (x, y); x += w + margin; shelf = max(shelf, h)
        return pos, y + shelf + margin <= 1
    lo, hi = 0.1, 10.0
    for _ in range(40):
        mid_ = (lo + hi) / 2
        lo, hi = (mid_, hi) if place(mid_)[1] else (lo, mid_)
    pos, _ = place(lo)
    out = UV.copy()
    for i in range(len(boxes)):
        m = tid == i
        out[m] = (UV[m] - boxes[i][0]) * lo + np.array(pos[i])
    # texel density report (2048 map): px per mm on the face
    a3 = np.linalg.norm(np.cross(V3[T[:, 1]] - V3[T[:, 0]], V3[T[:, 2]] - V3[T[:, 0]]), axis=1) / 2
    e1, e2 = out[:, 1] - out[:, 0], out[:, 2] - out[:, 0]
    a2 = np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) / 2
    print(f'UV islands {len(boxes)}  scale {lo:.2f}  median density @2048: {np.median(np.sqrt(a2 / np.maximum(a3, 1e-12))) * 2048 / 1000:.1f} px/mm')
    return out


def build():
    base = fetch('3dobjs/base.obj', os.path.join(MH, 'base.obj'))
    V0, VT, F, FT, G = load_obj(base)
    V = V0.copy()
    for rel, w in {**MACRO, **FACE}.items():
        V += w * load_target(rel, len(V0))
    eyeL, eyeR = group_center(V, F, G, 'joint-l-eye'), group_center(V, F, G, 'joint-r-eye')
    eyeL0, eyeR0 = group_center(V0, F, G, 'joint-l-eye'), group_center(V0, F, G, 'joint-r-eye')
    neck = group_center(V, F, G, 'joint-neck')
    head_j = group_center(V, F, G, 'joint-head')
    # ---- placement on the rig: eyes on the face photo's eye line, neck centred in the collar
    sys.path.insert(0, HERE)
    import parts as P
    ipd_hero = 2 * P.EYE_X
    s = ipd_hero / np.linalg.norm(eyeL - eyeR)
    mid = (eyeL + eyeR) / 2

    def place(X):
        Y = (X - mid) * s
        Y[:, 1] += P.Y_EYE
        Y[:, 2] += -0.0155 - (neck[2] - mid[2]) * s
        return Y
    # ---- head + neck cut out of the body
    body = [k for k, g in enumerate(G) if g == 'body']
    ear_w = np.zeros(len(V))
    for side in ('l', 'r'):
        try:
            D = np.linalg.norm(load_target(f'ears/{side}-ear-wing-incr', len(V)), axis=1)
            ear_w = np.maximum(ear_w, np.clip(D / (0.15 * D.max()), 0, 1))
        except Exception as e:
            print('ear weights', e)
    try:
        Dn = np.linalg.norm(load_target('nose/nose-scale-horiz-incr', len(V)), axis=1)
        nose_w = np.clip(Dn / (0.5 * Dn.max()), 0, 1)
    except Exception as e:
        print('nose weights', e); nose_w = None
    Vh = warp(place(V), ear_w, nose_w)
    NECK_Z = -0.008
    rn_all = np.sqrt(Vh[:, 0] ** 2 + (Vh[:, 2] - NECK_Z) ** 2)
    keep = []
    for k in body:
        f = F[k]
        ys = Vh[f, 1]
        # neck down into the tee collar; drop the shoulder/trapezius flare
        if ys.min() > 1.36 and not (ys.max() < 1.545 and rn_all[f].max() > 0.088 and np.abs(Vh[f, 0]).max() > 0.06):
            keep.append(k)
    # pull the neck base into a cylinder that stays inside the tee's rib collar
    t = np.clip((1.545 - Vh[:, 1]) / 0.05, 0, 1)
    rmax = 0.0615 + (1 - t) * 0.016
    rmin = rmax - 0.005                                   # and never thinner than the collar opening (no gap)
    sc = np.where((Vh[:, 1] < 1.545) & (rn_all > rmax), rmax / np.maximum(rn_all, 1e-9), 1.0)
    sc = np.where((Vh[:, 1] < 1.545) & (rn_all < rmin) & (rn_all > 0.02), rmin / np.maximum(rn_all, 1e-9), sc)
    Vh[:, 0] *= sc
    Vh[:, 2] = NECK_Z + (Vh[:, 2] - NECK_Z) * sc
    Fk = [F[k] for k in keep]
    T, TT = triangulate(Fk, [FT[k] for k in keep])
    used = np.unique(T)
    remap = -np.ones(len(V), int); remap[used] = np.arange(len(used))
    head_V = Vh[used]; head_F = remap[T]
    # ---- eyes: MakeHuman high-poly eyeballs moved with the morphed sockets
    eobj = fetch('eyes/high-poly/high-poly.obj', os.path.join(MH, 'eyes', 'high-poly.obj'))
    fetch('eyes/materials/brown_eye.png', os.path.join(MH, 'eyes', 'brown_eye.png'))
    EV, EVT, EF, EFT, EG = load_obj(eobj)
    ET, ETT = triangulate(EF, EFT)
    # drop the cornea shells (meant to be transparent; they map to the small
    # patch at the bottom-right of the eye texture)
    cuv = EVT[ETT]
    cornea = ((cuv[..., 0] > 0.8) & (cuv[..., 1] < 0.2)).all(1)
    print('eye faces', len(ET), 'cornea faces removed', int(cornea.sum()))
    ET, ETT = ET[~cornea], ETT[~cornea]
    EVm = fit_proxy(fetch('eyes/high-poly/high-poly.mhclo', os.path.join(MH, 'eyes', 'high-poly.mhclo')), V, len(EV))
    eyes_V = warp(place(EVm))
    inward = json.load(open(os.path.join(HERE, 'face_tweaks.json'))).get('eyeball_inward', 0.0)
    eyes_V[:, 0] -= np.sign(eyes_V[:, 0]) * inward          # critics: the irises sat ~2 mm too far out
    eb = json.load(open(os.path.join(HERE, 'face_tweaks.json'))).get('warp', {}).get('eyes_back', 0.0)
    if eb:
        for sg in (1, -1):
            E = eyes_V[eyes_V[:, 0] * sg > 0]
            c = E.mean(0)
            def wgt(X):
                r = np.linalg.norm((X - c) / [0.026, 0.020, 0.030], axis=1)
                return np.clip(1 - r, 0, 1) ** 1.5 * (X[:, 2] > c[2] - 0.02)
            head_V[:, 2] -= eb * wgt(head_V)
            m = eyes_V[:, 0] * sg > 0
            eyes_V[m, 2] -= eb                                         # eyeball moves rigidly with the lids
    # lower lids: drop the lid edge (critic: heavy-lidded, iris fully shown at the bottom) keeping the lid on the
    # eyeball; tapers to 0 at both eye corners and fades out ~8 mm below the edge
    drop = json.load(open(os.path.join(HERE, 'face_tweaks.json'))).get('warp', {}).get('lower_lid_drop', 0.0)
    if drop:
        for sg in (1, -1):
            E = eyes_V[eyes_V[:, 0] * sg > 0]
            c = E.mean(0); r = np.linalg.norm(E - c, axis=1).mean()
            rel = head_V - c
            u = np.abs(rel[:, 0]) / 0.015                                     # 0 at the pupil, 1 at the corners
            w = np.clip(1 - u ** 2, 0, 1) * _ss(0.001, -0.003, rel[:, 1]) * _ss(-0.016, -0.007, rel[:, 1]) * (rel[:, 2] > -0.002)
            d0 = np.linalg.norm(rel, axis=1) - r
            P2 = head_V.copy(); P2[:, 1] -= drop * w
            near = (d0 < 0.003) & (w > 0)
            rel2 = P2[near] - c
            P2[near] = c + rel2 / np.linalg.norm(rel2, axis=1, keepdims=True) * (r + d0[near])[:, None]
            head_V = P2
    # per-corner UVs for the eyes
    eyes_UV = EVT[ETT]
    # ---- landmarks along the facial midline (for aligning the face photo)
    mid_m = (np.abs(head_V[:, 0]) < 0.004) & (head_V[:, 2] > 0.02)
    prof = head_V[mid_m]
    prof = prof[np.argsort(-prof[:, 1])]
    ye = P.Y_EYE
    def zmax(y0, y1):
        m = (prof[:, 1] < y0) & (prof[:, 1] > y1); q = prof[m]; return q[np.argmax(q[:, 2])]
    tip = zmax(ye - 0.01, ye - 0.08)
    front = prof[:, 2] > tip[2] - 0.045            # outer surface only (skip the mouth cavity)

    def zmin(y0, y1):
        m = (prof[:, 1] < y0) & (prof[:, 1] > y1) & front; q = prof[m]; return q[np.argmin(q[:, 2])]
    subn = zmin(tip[1] - 0.004, tip[1] - 0.03)
    ulip = zmax(subn[1] - 0.004, subn[1] - 0.025)
    llip = zmax(ulip[1] - 0.008, ulip[1] - 0.035)
    stom = zmin(ulip[1], llip[1])
    pogo = zmax(llip[1] - 0.012, llip[1] - 0.06)
    under = prof[(prof[:, 1] < pogo[1]) & (prof[:, 1] > pogo[1] - 0.05)]
    # menton: lowest chin point still well in front of the neck
    ment = under[under[:, 2] > pogo[2] - 0.02]
    ment = ment[np.argmin(ment[:, 1])] if len(ment) else pogo
    glab = zmax(ye + 0.03, ye + 0.005)
    lm = {'eye_y': float(ye), 'eye_x': float(P.EYE_X), 'tip': tip.tolist(), 'subnasale': subn.tolist(), 'stomion': stom.tolist(),
          'pogonion': pogo.tolist(), 'menton': ment.tolist(), 'glabella': glab.tolist(),
          'eye_z': float(((eyeL - mid) * s)[2] - 0.0155 - (neck[2] - mid[2]) * s), 'scale': float(s),
          'head_joint': place(head_j[None])[0].tolist()}
    out = os.path.join(ROOT, 'build', 'cache')
    os.makedirs(out, exist_ok=True)
    head_UV = pack_uv(VT, TT, Vh, T)
    np.savez(os.path.join(out, 'mh_head.npz'), V=head_V, F=head_F, EV=eyes_V, EF=ET, EUV=eyes_UV, base_idx=used,
             xf=np.r_[s, mid, neck], UV=head_UV)
    json.dump(lm, open(os.path.join(out, 'mh_landmarks.json'), 'w'), indent=1)
    print('head', head_V.shape, head_F.shape, 'eyes', eyes_V.shape, 'scale', s)
    print(json.dumps(lm, indent=1))
    return head_V, head_F


if __name__ == '__main__':
    build()
