"""Bake procedural surface colours into a UV atlas: rasterise each triangle
in UV space, interpolate position/normal, call a paint function, then pad."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from texlib import dilate_colors, normal_from_albedo, save_rgb


def rasterize(UVt, size, supersample=1):
    """UVt: (m,3,2) per-triangle UVs in [0,1]. Returns texel ids, triangle ids
    and barycentrics for every covered texel (texel centres)."""
    S = size
    tex_ids, tri_ids, bars = [], [], []
    P = UVt * S - 0.5          # pixel-centre coordinates
    for t in range(len(P)):
        a, b, c = P[t]
        x0 = int(np.floor(min(a[0], b[0], c[0]))) - 1; x1 = int(np.ceil(max(a[0], b[0], c[0]))) + 1
        y0 = int(np.floor(min(a[1], b[1], c[1]))) - 1; y1 = int(np.ceil(max(a[1], b[1], c[1]))) + 1
        x0, y0 = max(x0, 0), max(y0, 0); x1, y1 = min(x1, S - 1), min(y1, S - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        xs = xs.ravel().astype(float); ys = ys.ravel().astype(float)
        det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(det) < 1e-12:
            continue
        l1 = ((b[1] - c[1]) * (xs - c[0]) + (c[0] - b[0]) * (ys - c[1])) / det
        l2 = ((c[1] - a[1]) * (xs - c[0]) + (a[0] - c[0]) * (ys - c[1])) / det
        l3 = 1 - l1 - l2
        eps = 0.6 / max(1.0, np.sqrt(abs(det)))     # generous edge inclusion (conservative raster)
        ok = (l1 >= -eps) & (l2 >= -eps) & (l3 >= -eps)
        if not ok.any():
            continue
        bc = np.clip(np.c_[l1[ok], l2[ok], l3[ok]], 0, None)
        bc /= bc.sum(1, keepdims=True)
        tex_ids.append((ys[ok].astype(int) * S + xs[ok].astype(int)))
        tri_ids.append(np.full(ok.sum(), t))
        bars.append(bc)
    tex = np.concatenate(tex_ids); tri = np.concatenate(tri_ids); bar = np.concatenate(bars)
    # keep the first triangle per texel (interior texels are unique anyway)
    _, first = np.unique(tex, return_index=True)
    return tex[first], tri[first], bar[first]


def bake(V, Nv, F, UVt, paint, size, flip_v=True, chunk=400_000, tri_attr=None):
    """V,Nv: vertex positions/normals; F: (m,3) vertex ids; UVt: (m,3,2) loop UVs.
    Returns (rgb, mask) with row 0 = top of the image (v flipped)."""
    tex, tri, bar = rasterize(UVt, size)
    Pt = (V[F[tri]] * bar[..., None]).sum(1)
    Nt = (Nv[F[tri]] * bar[..., None]).sum(1)
    Nt /= np.maximum(np.linalg.norm(Nt, axis=1, keepdims=True), 1e-9)
    col = np.zeros((len(tex), 3))
    ids = tri_attr[tri] if tri_attr is not None else None
    for i in range(0, len(tex), chunk):
        if ids is None:
            col[i:i + chunk] = paint(Pt[i:i + chunk], Nt[i:i + chunk])
        else:
            col[i:i + chunk] = paint(Pt[i:i + chunk], Nt[i:i + chunk], ids[i:i + chunk])
    img = np.zeros((size * size, 4), np.float32)
    img[tex, :3] = col
    img[tex, 3] = 1
    img = img.reshape(size, size, 4)
    if flip_v:
        img = img[::-1]
    rgb = dilate_colors(img, iters=16)
    return rgb, img[..., 3]


def write_maps(path_base, rgb, mask, normal_strength=None, detail_sigma=4.0, jpg=True):
    ext = '.jpg' if jpg else '.png'
    save_rgb(path_base + '_albedo' + ext, rgb)
    out = {'albedo': path_base + '_albedo' + ext}
    if normal_strength:
        n = normal_from_albedo(rgb, strength=normal_strength, detail_sigma=detail_sigma)
        save_rgb(path_base + '_normal.jpg', n, quality=93)
        out['normal'] = path_base + '_normal.jpg'
    return out
