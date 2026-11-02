"""Generate (and cache) raw marching-cubes meshes for each part.
    python geometry_cache.py [part ...]   (no args = all parts)"""
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import parts as P
from meshes import mc, mirror_mesh, remove_cap_faces, skirt_grid
from sdf import union

CACHE = os.path.join(P.ROOT, 'build', 'cache')
os.makedirs(CACHE, exist_ok=True)


def gen(name):
    print('part', name)
    if name == 'head':
        # MakeHuman head (base mesh + morph targets), see makehuman_head.py
        from makehuman_head import build as mh_build
        V, F = mh_build()
    elif name == 'ear_l':
        V, F = mc(P.ear_sdf_left, P.EAR_BOX, 0.0007)
    elif name == 'hair':
        V, F = mc(P.hair_sdf, P.HAIR_BOX, 0.0006)
    elif name == 'bun':
        V, F = mc(P.bun_sdf, P.BUN_BOX, 0.001)
    elif name == 'tee':
        V, F = mc(lambda p: union(P.tee_sdf(p), P.collar_rib_sdf(p), k=0.003), P.TEE_BOX, 0.003)
        F = remove_cap_faces(V, F, 1, P.TEE_HEM, -1)
    elif name == 'jeans':
        V, F = mc(P.jeans_sdf, P.JEANS_BOX, 0.0028)
        F = remove_cap_faces(V, F, 1, P.JEANS_HEM, -1)
        F = remove_cap_faces(V, F, 1, P.JEANS_TOP, 1)
    elif name == 'shoe_l':
        V, F = mc(P.shoe_sdf_left, P.SHOE_BOX, 0.0014)
    elif name == 'hand_l':
        V, F = mc(P.hand_sdf_left, P.HAND_BOX, 0.0011)
    elif name == 'coat':
        V, F = mc(P.coat_sdf, P.COAT_BOX, 0.0021)
    elif name == 'collar':
        V, F = mc(P.coat_collar_sdf, ((-0.135, 1.455, -0.15), (0.135, 1.535, 0.11)), 0.0011)
    elif name == 'cuff_l':
        V, F = mc(P.cuff_sdf_left, P.CUFF_BOX, 0.0012)
    elif name == 'skirt':
        V, F, meta = skirt_grid()
        with open(os.path.join(CACHE, 'skirt.json'), 'w') as f:
            json.dump(meta, f)
    else:
        raise KeyError(name)
    np.savez(os.path.join(CACHE, name + '.npz'), V=V, F=F)
    return V, F


ALL = ['head', 'hair', 'ear_l', 'bun', 'tee', 'jeans', 'shoe_l', 'hand_l', 'coat', 'collar', 'cuff_l', 'skirt']


def load(name):
    path = os.path.join(CACHE, name + '.npz')
    if not os.path.exists(path):
        gen(name)
    d = np.load(path)
    return d['V'], d['F']


if __name__ == '__main__':
    names = sys.argv[1:] or ALL
    for n in names:
        gen(n)
