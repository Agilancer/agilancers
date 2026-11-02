"""Detailed facial landmarks on the MakeHuman head (bind-pose hero space)
for the thin-plate-spline warp that aligns the face photo."""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))


def detect():
    d = np.load(os.path.join(ROOT, 'build/cache/mh_head.npz'))
    V, EV = d['V'], d['EV']
    lm = json.load(open(os.path.join(ROOT, 'build/cache/mh_landmarks.json')))
    out = {}
    for side, s in (('l', 1), ('r', -1)):
        E = EV[EV[:, 0] * s > 0]
        c = E.mean(0); r = np.linalg.norm(E - c, axis=1).mean()
        dist = np.linalg.norm(V - c, axis=1)
        rim = V[(np.abs(dist - r) < 0.0035) & (V[:, 2] > c[2] - 0.002) & (np.abs(V[:, 1] - c[1]) < 0.012)]
        inner = rim[np.argmin(np.abs(rim[:, 0]))]; outer = rim[np.argmax(np.abs(rim[:, 0]))]
        top = rim[np.argmax(rim[:, 1])]; bot = rim[np.argmin(rim[:, 1])]
        out[f'eye_{side}'] = c.tolist()
        out[f'canthus_in_{side}'] = inner.tolist(); out[f'canthus_out_{side}'] = outer.tolist()
        out[f'lid_top_{side}'] = top.tolist(); out[f'lid_bot_{side}'] = bot.tolist()
    # mouth corners: lateral end of the lip-seam groove
    sy = lm['stomion'][1]
    best = 0.0
    for xb in np.arange(0.004, 0.04, 0.001):
        m = (np.abs(V[:, 0] - xb) < 0.0008) & (np.abs(V[:, 1] - sy) < 0.009) & (V[:, 2] > lm['stomion'][2] - 0.03)
        q = V[m]
        if len(q) < 4: continue
        seam = q[np.abs(q[:, 1] - sy) < 0.0018]
        up = q[(q[:, 1] - sy > 0.003)]; dn = q[(sy - q[:, 1] > 0.003)]
        if len(seam) and len(up) and len(dn):
            depth = 0.5 * (up[:, 2].max() + dn[:, 2].max()) - seam[:, 2].max()
            if depth > 0.0008: best = xb
    best = min(best, 0.0265)            # groove detection can run into the nasolabial folds
    for side, s in (('l', 1), ('r', -1)):
        m = (np.abs(V[:, 0] - s * best) < 0.001) & (np.abs(V[:, 1] - sy) < 0.003) & (V[:, 2] > 0.05)
        q = V[m]; p = q[np.argmax(q[:, 2])] if len(q) else np.array([s * best, sy, lm['stomion'][2] - 0.01])
        out[f'mouth_{side}'] = p.tolist()
    # nose wings (alae)
    ny = lm['subnasale'][1] + 0.004
    m = (np.abs(V[:, 1] - ny) < 0.002) & (V[:, 2] > lm['subnasale'][2] - 0.012)
    q = V[m & (np.abs(V[:, 0]) < 0.019)]
    for side, s in (('l', 1), ('r', -1)):
        qq = q[q[:, 0] * s > 0]
        out[f'ala_{side}'] = qq[np.argmax(np.abs(qq[:, 0]))].tolist()
    # jaw angle (gonion): widest point of the lower face below the mouth
    m = (V[:, 1] < sy - 0.005) & (V[:, 1] > lm['menton'][1] + 0.005) & (V[:, 2] > -0.02)
    q = V[m]
    for side, s in (('l', 1), ('r', -1)):
        qq = q[q[:, 0] * s > 0]
        out[f'gonion_{side}'] = qq[np.argmax(np.abs(qq[:, 0]))].tolist()
    lm.update(out)
    json.dump(lm, open(os.path.join(ROOT, 'build/cache/mh_landmarks.json'), 'w'), indent=1)
    return lm


def from_vertices():
    """Landmarks from fixed MakeHuman vertex patches (chosen once on a clean
    head by fit_face.py) - robust to any face morphing."""
    path = os.path.join(ROOT, 'build/cache/face_landmark_vertices.json')
    d = np.load(os.path.join(ROOT, 'build/cache/mh_head.npz'))
    V, bidx = d['V'], d['base_idx']
    pos = {int(b): i for i, b in enumerate(bidx)}
    lm = json.load(open(os.path.join(ROOT, 'build/cache/mh_landmarks.json')))
    patches = json.load(open(path))['patch']
    for k, pl in patches.items():
        ii = [pos[b] for b in pl if b in pos]
        if ii:
            lm[k] = V[ii].mean(0).tolist()
    EV = d['EV']
    for side, sg in (('l', 1), ('r', -1)):
        lm[f'eye_{side}'] = EV[EV[:, 0] * sg > 0].mean(0).tolist()
    json.dump(lm, open(os.path.join(ROOT, 'build/cache/mh_landmarks.json'), 'w'), indent=1)
    return lm


if __name__ == '__main__':
    use_fixed = os.path.exists(os.path.join(ROOT, 'build/cache/face_landmark_vertices.json')) and '--detect' not in sys.argv
    lm = from_vertices() if use_fixed else detect()
    for k, v in lm.items():
        print(k, np.round(v, 4) if isinstance(v, list) else v)
