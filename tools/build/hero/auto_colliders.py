"""Fit the cloth colliders to the skinned under-layers (tee + jeans) and
write them into src/cloth/topology.js."""
import os, sys, re
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import parts as P
from geometry_cache import load

tV, _ = load('tee'); jV, _ = load('jeans')
under = np.concatenate([tV[np.abs(tV[:, 0]) < 0.205], jV[jV[:, 1] > 0.12]])


def seg_dist(p, a, b):
    ba = b - a; pa = p - a
    h = np.clip(pa @ ba / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1)


axes = []
# torso: horizontal capsules stacked along the spine
for y, bone in [(1.46, 'spine_03'), (1.40, 'spine_03'), (1.34, 'spine_03'), (1.28, 'spine_02'), (1.22, 'spine_02'),
                (1.16, 'spine_02'), (1.10, 'spine_01'), (1.04, 'spine_01'), (0.98, 'pelvis'), (0.92, 'pelvis')]:
    m = (np.abs(under[:, 1] - y) < 0.02) & (np.abs(under[:, 0]) < 0.2)
    q = under[m]
    hw = np.abs(q[:, 0]).max(); z0, z1 = q[:, 2].min(), q[:, 2].max()
    hd = (z1 - z0) / 2; zc = (z0 + z1) / 2
    hx = max(hw - hd, 0.01)
    axes.append([bone, np.array([-hx, y, zc]), np.array([hx, y, zc])])
# legs
for s, side in ((1, 'l'), (-1, 'r')):
    for (y0, y1, bone) in [(0.88, 0.55, f'thigh_{side}'), (0.53, 0.16, f'calf_{side}')]:
        pts = []
        for y in (y0, y1):
            m = (np.abs(under[:, 1] - y) < 0.02) & (under[:, 0] * s > 0.01)
            q = under[m]
            pts.append(np.array([(q[:, 0].min() + q[:, 0].max()) / 2, y, (q[:, 2].min() + q[:, 2].max()) / 2]))
        axes.append([bone, pts[0], pts[1]])
# assign every under-layer vertex to its nearest axis -> radius = 97th percentile
D = np.stack([seg_dist(under, a, b) for _, a, b in axes], 1)
own = np.argmin(D, 1)
out = []
for k, (bone, a, b) in enumerate(axes):
    d = D[own == k, k]
    r = float(np.percentile(d, 97)) if len(d) else 0.05
    if 'thigh' in bone: r = min(r, 0.098)       # crotch/knee verts inflate the fit
    if 'calf' in bone: r = min(r, 0.072)
    out.append((bone, a, b, r))
# arms (coat sleeves / forearms)
out += [('upperarm_l', np.array([0.225, 1.442, -0.065]), np.array([0.455, 1.441, -0.070]), 0.066),
        ('upperarm_r', np.array([-0.225, 1.442, -0.065]), np.array([-0.455, 1.441, -0.070]), 0.066),
        ('lowerarm_l', np.array([0.480, 1.441, -0.070]), np.array([0.725, 1.441, -0.066]), 0.045),
        ('lowerarm_r', np.array([-0.480, 1.441, -0.070]), np.array([-0.725, 1.441, -0.066]), 0.045)]
lines = []
for bone, a, b, r in out:
    lines.append(f"  ['{bone}', {a[0]:.3f}, {a[1]:.3f}, {a[2]:.3f}, {b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {r:.3f}],")
js = '\n'.join(lines)
p = os.path.join(P.ROOT, 'src/cloth/topology.js')
s = open(p).read()
s = re.sub(r"export const COLLIDERS = \[\n.*?\n\];", "export const COLLIDERS = [\n" + js + "\n];", s, flags=re.S)
open(p, 'w').write(s)
print(js)
