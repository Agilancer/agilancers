"""Check how well the cloth capsule colliders (src/cloth/topology.js) enclose
the skinned under-layers (tee + jeans) in the bind pose."""
import os, sys, re, json
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import parts as P
from geometry_cache import load

src = open(os.path.join(P.ROOT, 'src/cloth/topology.js')).read()
rows = re.findall(r"\['(\w+)',\s*([-\d.,\s]+)\]", src.split('COLLIDERS = [')[1])
caps = [(b, np.array([float(x) for x in nums.split(',') if x.strip()])) for b, nums in rows]


def cap_dist(p, c):
    a, b, r = c[:3], c[3:6], c[6]
    ba = b - a; pa = p - a
    h = np.clip(pa @ ba / (ba @ ba), 0, 1)
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - r


def report(name, V, mask):
    V = V[mask]
    d = np.min(np.stack([cap_dist(V, c) for _, c in caps], 1), 1)
    out = d > 0.004
    print(f'{name}: {len(V)} verts, outside>{4}mm: {out.mean()*100:.1f}%  max {d.max()*1000:.0f}mm')
    if out.any():
        for y0 in np.arange(0.4, 1.55, 0.05):
            m = out & (np.abs(V[:, 1] - y0) < 0.025)
            if m.sum() > 3:
                q = V[m]
                print(f'   y~{y0:.2f}: n={m.sum():4d} worst {d[m].max()*1000:.0f}mm  x[{q[:,0].min():+.2f},{q[:,0].max():+.2f}] z[{q[:,2].min():+.2f},{q[:,2].max():+.2f}]')


tV, _ = load('tee'); jV, _ = load('jeans')
report('tee (torso)', tV, np.abs(tV[:, 0]) < 0.20)
report('jeans', jV, jV[:, 1] > 0.12)
