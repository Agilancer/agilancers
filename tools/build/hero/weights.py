"""Skin weights: transferred from the original UAL2 mannequin (artist
weights) by inverse-distance blending of nearby mannequin vertices, then
Laplacian-smoothed over each new mesh and pruned to 4 influences."""
import os, sys
import numpy as np
from scipy.spatial import cKDTree
from scipy import sparse
sys.path.insert(0, os.path.dirname(__file__))
import parts as P

_m = np.load(os.path.join(P.ROOT, 'build/mannequin.npz'))
MV, MW = _m['V'], _m['W']
BONES = list(_m['bones'])
_tree = cKDTree(MV)


def bone_index(name):
    return BONES.index(name)


def transfer(V, k=10, allowed=None):
    d, idx = _tree.query(V, k=k)
    w = 1.0 / np.maximum(d, 1e-4) ** 2
    W = (MW[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    if allowed is not None:
        mask = np.zeros(len(BONES), bool)
        mask[[bone_index(b) for b in allowed]] = True
        W[:, ~mask] = 0
    return normalize(W)


def normalize(W):
    s = W.sum(1, keepdims=True)
    return W / np.maximum(s, 1e-9)


def adjacency(n, F):
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    A = sparse.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(1)).ravel()
    return sparse.diags(1.0 / np.maximum(deg, 1)) @ A


def smooth(W, F, iters=6, lam=0.5):
    L = adjacency(len(W), F)
    for _ in range(iters):
        W = (1 - lam) * W + lam * (L @ W)
    return normalize(W)


def limit(W, n=4, min_w=0.01):
    W = W.copy()
    order = np.argsort(-W, axis=1)
    keep = np.zeros_like(W, bool)
    rows = np.arange(len(W))[:, None]
    keep[rows, order[:, :n]] = True
    W[~keep] = 0
    W[W < min_w] = 0
    return normalize(W)


def rigid(V, bone):
    W = np.zeros((len(V), len(BONES)))
    W[:, bone_index(bone)] = 1
    return W


def blend_by(V, bones, coord, stops):
    """Piecewise blend between bones along a coordinate (e.g. height)."""
    W = np.zeros((len(V), len(BONES)))
    c = coord(V)
    for i in range(len(bones)):
        y = np.zeros(len(bones)); y[i] = 1
        W[:, bone_index(bones[i])] = np.interp(c, stops, y)
    return normalize(W)


def hand_weights(V, side='l'):
    """Fingers/palm from the hand primitives (each belongs to a bone); the
    forearm part blends with the mannequin transfer near the elbow."""
    Vl = V.copy()
    if side == 'r':
        Vl[:, 0] *= -1
    ds = np.stack([fn(Vl) for _, fn in P.HAND_PARTS], 1)
    names = [b if side == 'l' else b[:-2] + '_r' for b, _ in P.HAND_PARTS]
    d = np.maximum(ds, 0) + 0.0015
    w = 1.0 / d ** 4
    W = np.zeros((len(V), len(BONES)))
    for j, b in enumerate(names):
        W[:, bone_index(b)] += w[:, j]
    W = normalize(W)
    # forearm: keep elbow/wrist blending from the artist weights
    T = transfer(V, allowed=[f'upperarm_{side}', f'lowerarm_{side}', f'hand_{side}'])
    x = np.abs(V[:, 0])
    elbow = P.HEAD['lowerarm_l'][0]
    fore = np.clip((P.HEAD['hand_l'][0] - 0.01 - x) / 0.03, 0, 1)   # 1 on the forearm, 0 in the hand
    W = normalize(W * (1 - fore[:, None]) + T * fore[:, None])
    return W
