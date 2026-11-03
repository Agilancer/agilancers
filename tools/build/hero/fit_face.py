"""Fit MakeHuman face-shape morphs to the client's measured proportions.

Every MakeHuman target is a linear per-vertex offset, so landmark positions
are linear in the target weights. We solve a bounded, lightly regularised
least-squares problem (weights in [0, 1], left/right targets tied) so the
landmarks match those measured on the client's frontal (x, y) and profile
(depth) video frames. Output: build/cache/face_fit.json (used by
makehuman_head.py)."""
import os, sys, json
import numpy as np
from scipy.optimize import lsq_linear
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import makehuman_head as MH

ROOT = MH.ROOT
LAMBDA = float(os.environ.get('FIT_LAMBDA', '0.002'))   # metres of residual per unit of weight

# measured on the client's video frames (metres, relative to the eye midpoint; +x = his left, +y up)
FRONT = {   # name: (x, y)
    'canthus_in_l': (0.0179, -0.001), 'canthus_in_r': (-0.0179, -0.001),
    'canthus_out_l': (0.0468, 0.0), 'canthus_out_r': (-0.0468, 0.0),
    'eye_l': (0.0315, 0.0), 'eye_r': (-0.0315, 0.0),
    'brow_in_l': (0.0145, 0.0187), 'brow_in_r': (-0.0145, 0.0187),
    'brow_out_l': (0.0552, 0.0170), 'brow_out_r': (-0.0552, 0.0170),
    'tip': (0.0, -0.0285), 'subnasale': (0.0, -0.0459),
    'ala_l': (0.0183, -0.0423), 'ala_r': (-0.0183, -0.0423),
    'mouth_l': (0.0303, -0.0658), 'mouth_r': (-0.0303, -0.0658), 'stomion': (0.0, -0.0658),
    'menton': (0.0, -0.1127),
    'gonion_l': (0.0698, -0.0686), 'gonion_r': (-0.0698, -0.0686),
}
PROFILE = {  # name: (depth ahead of subnasale, weight)
    'tip': (0.0135, 1.0), 'nasion': (-0.0038, 0.8), 'glabella': (0.0, 0.6),
    'ulip': (0.0038, 0.8), 'llip': (0.0038, 0.8), 'stomion': (0.0, 0.6),
    'pogonion': (0.0080, 0.5), 'menton': (0.0040, 0.4),
}


BROAD = {
    'forehead': ('forehead-trans', 'forehead-scale-vert'),
    'eyebrows': ('eyebrows-trans',),
    'nose': ('nose-trans', 'nose-scale-vert', 'nose-scale-horiz', 'nose-scale-depth', 'nose-point-width', 'nose-width1',
             'nose-width2', 'nose-width3', 'nose-curve', 'nose-greek', 'nose-hump', 'nose-volume', 'nose-base'),
    'mouth': ('mouth-scale-horiz', 'mouth-scale-vert', 'mouth-scale-depth', 'mouth-trans', 'mouth-upperlip-volume',
              'mouth-lowerlip-volume'),
    'chin': ('chin-prominent', 'chin-width', 'chin-height', 'chin-jaw-drop', 'chin-bones', 'chin-prognathism'),
    'cheek': ('r-cheek-volume', 'r-cheek-bones', 'r-cheek-inner', 'r-cheek-trans'),
}


TWEAKS = json.load(open(os.path.join(HERE, 'face_tweaks.json')))
LAMBDA = float(TWEAKS.get('lambda', LAMBDA))
for k, v in TWEAKS.get('front', {}).items():
    FRONT[k] = tuple(v)
for k, v in TWEAKS.get('profile', {}).items():
    PROFILE[k] = tuple(v)
# absolute depths ahead of the eyeball centres (client profile frames, scaled by nose length): without these
# every depth is relative to the subnasale, which sits on the nose itself, so the solver can push the whole
# nose back into the face and still satisfy them
PROFILE_EYE = {k: tuple(v) for k, v in TWEAKS.get('profile_eye', {}).items()}
FRONT_W = TWEAKS.get('front_weight', {})


def candidates():
    mods = json.load(open(os.path.join(MH.MH, 'modeling_modifiers.json')))
    out = []
    for g in mods:
        grp = g['group']
        if grp not in ('head', 'forehead', 'eyebrows', 'eyes', 'nose', 'mouth', 'chin', 'cheek'):
            continue
        for m in g['modifiers']:
            t = m.get('target')
            if not t:
                continue
            if grp == 'head' and t in ('head-age', 'head-angle', 'head-trans', 'head-back-scale-depth'):
                continue
            if grp == 'eyes' and not any(k in t for k in ('eye-trans', 'eye-scale')):
                continue
            # only broad shape controls: small local targets can yank single landmark vertices into spikes
            if t in BROAD.get(grp, ()) or (grp in ('head',) and t not in ('head-age', 'head-angle', 'head-trans', 'head-back-scale-depth')):
                pass
            elif grp != 'eyes':
                continue
            ends = [m['min'], m['max']] if 'min' in m else [None]
            for e in ends:
                name = f'{t}-{e}' if e else t
                if name.startswith('l-'):
                    continue                       # tied to its r- twin below
                files = [f'{grp}/{name}']
                if name.startswith('r-'):
                    files.append(f'{grp}/l-{name[2:]}')
                if files[0] not in [o[0] for o in out]:
                    out.append((files[0], files))
    return out


def main():
    base = MH.fetch('3dobjs/base.obj', os.path.join(MH.MH, 'base.obj'))
    V0, VT, F, FT, G = MH.load_obj(base)
    n = len(V0)
    Vm = V0.copy()
    for rel, w in MH.MACRO.items():
        Vm += w * MH.load_target(rel, n)
    # the hand-set morph additions are part of the base the fit works around (not stacked on afterwards)
    for rel, w in TWEAKS.get('morph_add', {}).items():
        Vm += w * MH.load_target(rel, n)
    # landmark vertices (base-mesh indices) from the current head
    d = np.load(os.path.join(ROOT, 'build/cache/mh_head.npz'))
    HV, bidx = d['V'], d['base_idx']
    lm = json.load(open(os.path.join(ROOT, 'build/cache/mh_landmarks.json')))
    ey = lm['eye_y']

    def nearest(p):
        return int(bidx[np.argmin(np.linalg.norm(HV - np.asarray(p), axis=1))])

    def front_vertex(x, y):
        m = (np.abs(HV[:, 0] - x) < 0.004) & (np.abs(HV[:, 1] - y) < 0.004)
        k = np.nonzero(m)[0]
        return int(bidx[k[np.argmax(HV[k, 2])]])
    # midline profile points for lips
    mid = HV[(np.abs(HV[:, 0]) < 0.004) & (HV[:, 2] > lm['tip'][2] - 0.045)]
    def zmax(y0, y1):
        q = mid[(mid[:, 1] < y0) & (mid[:, 1] > y1)]; return q[np.argmax(q[:, 2])]
    def zmin(y0, y1):
        q = mid[(mid[:, 1] < y0) & (mid[:, 1] > y1)]; return q[np.argmin(q[:, 2])]
    pts = {k: nearest(lm[k]) for k in ('canthus_in_l', 'canthus_in_r', 'canthus_out_l', 'canthus_out_r', 'tip', 'subnasale',
                                       'ala_l', 'ala_r', 'mouth_l', 'mouth_r', 'stomion', 'menton', 'gonion_l', 'gonion_r',
                                       'pogonion', 'glabella')}
    pts['ulip'] = nearest(zmax(lm['subnasale'][1] - 0.003, lm['stomion'][1] + 0.001))
    pts['llip'] = nearest(zmax(lm['stomion'][1] - 0.001, lm['pogonion'][1] + 0.008))
    pts['nasion'] = nearest(zmin(lm['glabella'][1], lm['tip'][1] + 0.01))
    for s_, sg in (('l', 1), ('r', -1)):
        pts[f'brow_in_{s_}'] = front_vertex(sg * 0.016, ey + 0.0165)
        pts[f'brow_out_{s_}'] = front_vertex(sg * 0.050, ey + 0.0128)
    eyeL = sorted({i for f, g in zip(F, G) if g == 'joint-l-eye' for i in f})
    eyeR = sorted({i for f, g in zip(F, G) if g == 'joint-r-eye' for i in f})
    scale = float(d['xf'][0])

    LV = os.path.join(ROOT, 'build/cache/face_landmark_vertices.json')
    if os.path.exists(LV):                               # keep the vertex choice stable across refits
        pts = {k: int(v) for k, v in json.load(open(LV))['vertex'].items()}
    patches = {}
    for k, i in pts.items():
        dd = np.linalg.norm(Vm - Vm[i], axis=1)
        rad = 0.014 if k in ('stomion', 'mouth_l', 'mouth_r', 'canthus_in_l', 'canthus_in_r', 'canthus_out_l', 'canthus_out_r') else 0.035
        patches[k] = np.nonzero(dd < rad)[0]            # ~1.5 / 3.5 mm patch on the 1/10-scale MakeHuman mesh
    json.dump({'vertex': pts, 'patch': {k: v.tolist() for k, v in patches.items()}}, open(LV, 'w'))

    def landmark_matrix(Vx):
        """rows: landmark coords relative to the eye midpoint (hero metres)."""
        em = 0.5 * (Vx[eyeL].mean(0) + Vx[eyeR].mean(0))
        out = {k: (Vx[patches[k]].mean(0) - em) * scale for k in pts}
        out['eye_l'] = (Vx[eyeL].mean(0) - em) * scale
        out['eye_r'] = (Vx[eyeR].mean(0) - em) * scale
        return out

    def residual_vec(L):
        r = []
        for k, (x, y) in FRONT.items():
            fw = FRONT_W.get(k, 1.0)
            r += [fw * (L[k][0] - x), fw * (L[k][1] - y)]
        for k, (z, w) in PROFILE.items():
            r += [w * ((L[k][2] - L['subnasale'][2]) - z)]
        for k, (z, w) in PROFILE_EYE.items():                # depth ahead of the eyeball centres
            r += [w * (L[k][2] - z)]
        return np.array(r)

    L0 = landmark_matrix(Vm)
    r0 = residual_vec(L0)
    cands = candidates()
    cols, names = [], []
    for key, files in cands:
        D = np.zeros_like(Vm)
        try:
            for f in files:
                D += MH.load_target(f, n)
        except Exception as e:
            print('skip', key, e); continue
        Lk = landmark_matrix(Vm + D)
        cols.append(residual_vec(Lk) - r0)
        names.append((key, files))
    A = np.array(cols).T
    # regularised: [A; lambda*I] w = [-r0; 0]
    # whole-head presets and eye targets are strong/overlapping: regularise them harder
    lam = np.array([LAMBDA * (3.0 if k.split('/')[0] in ('head', 'eyes') else 1.0) for k, _ in names])
    Areg = np.vstack([A, np.diag(lam)])
    breg = np.r_[-r0, np.zeros(A.shape[1])]
    # per-target caps (face_tweaks 'caps'): keep the solver off targets that read badly at full strength
    caps = TWEAKS.get('caps', {})
    ub = np.array([max(caps.get(k, 1.0), 1e-6) for k, _ in names])
    sol = lsq_linear(Areg, breg, bounds=(np.zeros(len(ub)), ub), lsmr_tol='auto')
    w = sol.x
    r1 = r0 + A @ w
    print(f'targets {len(names)}  rms before {np.sqrt((r0**2).mean())*1000:.2f} mm  after {np.sqrt((r1**2).mean())*1000:.2f} mm')
    labels = [f'{k}.{c}' for k in FRONT for c in 'xy'] + [f'{k}.z' for k in PROFILE] + [f'{k}.zeye' for k in PROFILE_EYE]
    for lab, a, b in zip(labels, r0, r1):
        if abs(a) > 0.002 or abs(b) > 0.002:
            print(f'   {lab:18s} {a*1000:+6.1f} -> {b*1000:+6.1f} mm')
    weights = {}
    for (key, files), wi in zip(names, w):
        if wi > 1e-3:
            for f in files:
                weights[f] = round(float(wi), 4)
    top = sorted(((v, k) for k, v in weights.items() if not k.split('/')[1].startswith('l-')), reverse=True)
    print('active:', ', '.join(f'{k.split("/")[1]}={v:.2f}' for v, k in top[:30]))
    json.dump({'weights': weights, 'rms_mm': float(np.sqrt((r1 ** 2).mean()) * 1000), 'lambda': LAMBDA},
              open(os.path.join(ROOT, 'build/cache/face_fit.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
