"""Assemble the Agilancer hero in Blender and export a GLB with all UAL2
animations.  Run with the bpy venv (see tools/build/README.md):

    python tools/build/hero/build_hero.py [--preview]

--preview skips texture baking (flat colours) for fast geometry checks."""
import os, sys, json, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bpy
import parts as P
import weights as WT
import paint as PT
from geometry_cache import load, CACHE
from blender_util import *
from bake import bake, write_maps

PREVIEW = '--preview' in sys.argv
OUT = os.path.join(P.ROOT, 'build/hero_preview.glb' if PREVIEW else 'assets/models/agilancer_hero.glb')
TEXDIR = os.path.join(P.ROOT, 'build', 'textures')
os.makedirs(TEXDIR, exist_ok=True)

t0 = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.join(P.ROOT, 'assets/models/UAL2_Standard.glb'))
arm = bpy.data.objects['Armature']
for name in ['Mannequin', 'Icosphere']:
    o = bpy.data.objects.get(name)
    if o:
        bpy.data.objects.remove(o, do_unlink=True)
arm.name = 'Agilancer'
BONES = WT.BONES


def log(*a):
    print(*a, f'({time.time()-t0:.0f}s)', flush=True)


def mirror(V, F):
    V2 = V.copy(); V2[:, 0] *= -1
    return V2, F[:, ::-1].copy()


def sub(name, V, F, target, wfn=None, subdiv=0):
    """Decimate (or Catmull-Clark smooth) one sub-part and compute its skin weights."""
    obj = make_mesh('_tmp_' + name, V, F)
    if subdiv:
        import blender_util as BU
        m = obj.modifiers.new('cc', 'SUBSURF'); m.levels = subdiv; m.render_levels = subdiv
        m.boundary_smooth = 'PRESERVE_CORNERS'
        BU.apply_modifiers(obj)
        triangulate(obj)
    decimate(obj, target)
    triangulate(obj)
    V2, F2 = mesh_arrays(obj)
    bpy.data.objects.remove(obj, do_unlink=True)
    if wfn is None:
        W = WT.limit(WT.smooth(WT.transfer(V2), F2))
    else:
        W = wfn(V2, F2)
    return dict(name=name, V=V2, F=F2, W=W)


def vertex_ao(V, F, extra=(), rays=32, max_dist=0.04, strength=0.85):
    """Per-vertex ambient occlusion (cosine-weighted hemisphere rays against
    this mesh plus optional occluder meshes)."""
    from mathutils.bvhtree import BVHTree
    from mathutils import Vector
    allV, allF, off = [V], [F], len(V)
    for (EV, EF) in extra:
        allV.append(EV); allF.append(EF + off); off += len(EV)
    PV = np.concatenate(allV); PF = np.concatenate(allF)
    bvh = BVHTree.FromPolygons([tuple(v) for v in PV.tolist()], [tuple(f) for f in PF.tolist()])
    N = np_vertex_normals(V, F)
    rng = np.random.default_rng(3)
    u1, u2 = rng.random(rays), rng.random(rays)
    dirs = np.c_[np.sqrt(u1) * np.cos(2 * np.pi * u2), np.sqrt(u1) * np.sin(2 * np.pi * u2), np.sqrt(1 - u1)]
    ao = np.ones(len(V))
    for i in range(len(V)):
        n = N[i]
        t = np.cross(n, [0, 1, 0] if abs(n[1]) < 0.9 else [1, 0, 0]); t /= np.linalg.norm(t)
        b = np.cross(n, t)
        o = V[i] + n * 0.0006
        hit = 0
        for d in dirs:
            w = d[0] * t + d[1] * b + d[2] * n
            loc, nor, idx, dist = bvh.ray_cast(Vector(o), Vector(w), max_dist)
            if loc is not None:
                hit += 1
        ao[i] = 1 - strength * hit / rays
    return ao


def ao_paint(paint, V, ao):
    from scipy.spatial import cKDTree
    tree = cKDTree(V)
    def f(Pt, N, *rest):
        col = paint(Pt, N, *rest)
        d, idx = tree.query(Pt, k=4)
        w = 1 / np.maximum(d, 1e-5)
        a = (ao[idx] * w).sum(1) / w.sum(1)
        col = col * a[:, None]
        lum = col.mean(1, keepdims=True)                 # occlusion greys, it doesn't redden
        return col + (lum - col) * (0.5 * (1 - a))[:, None]
    return f


def assemble(name, subs, paint=None, size=1024, rough=0.7, metal=0.0, normal=None, double_sided=False, detail_sigma=4.0, color=(0.5, 0.5, 0.5), ao=None, orm_paint=None):
    Vs, Fs, Ws, ids = [], [], [], []
    off = 0
    for i, s in enumerate(subs):
        Vs.append(s['V']); Fs.append(s['F'] + off); Ws.append(s['W']); ids.append(np.full(len(s['F']), i))
        off += len(s['V'])
    V, F, W, tid = np.concatenate(Vs), np.concatenate(Fs), np.concatenate(Ws), np.concatenate(ids)
    obj = make_mesh(name, V, F)
    REBAKE = os.environ.get('REBAKE')
    cached = os.path.join(TEXDIR, name + '_albedo.jpg')
    if REBAKE and name not in REBAKE.split(',') and os.path.exists(cached) and not PREVIEW and paint is not None:
        smart_uv(obj)
        nrm = os.path.join(TEXDIR, name + '_normal.jpg')
        orm = os.path.join(TEXDIR, name + '_orm.png')
        mat = image_material('M_' + name, cached, nrm if os.path.exists(nrm) else None, rough, metal, double_sided=double_sided,
                             orm_path=orm if (orm_paint is not None and os.path.exists(orm)) else None)
    elif PREVIEW or paint is None:
        mat = flat_material('M_' + name, color, rough, metal)
        mat.use_backface_culling = not double_sided
    else:
        smart_uv(obj)
        uv = get_loop_uv(obj)
        LT = loop_tris(obj)
        N = np_vertex_normals(V, F)
        if ao is not None:
            t_ao = time.time()
            aov = vertex_ao(V, F, **ao)
            paint = ao_paint(paint, V, aov)
            log(f'  ao {name}: {time.time()-t_ao:.0f}s')
        rgb, mask = bake(V, N, F, uv[LT], paint, size, tri_attr=tid)
        maps = write_maps(os.path.join(TEXDIR, name), rgb, mask, normal_strength=normal, detail_sigma=detail_sigma)
        orm = None
        if orm_paint is not None:                      # per-texel roughness (G) / metal (B)
            orgb, _ = bake(V, N, F, uv[LT], orm_paint, size // 2, tri_attr=tid)
            orm = os.path.join(TEXDIR, name + '_orm.png')
            from PIL import Image as _I
            _I.fromarray((np.clip(orgb, 0, 1) * 255).astype(np.uint8)).save(orm)
        mat = image_material('M_' + name, maps['albedo'], maps.get('normal'), rough, metal, double_sided=double_sided, orm_path=orm)
    obj.data.materials.append(mat)
    skin(obj, arm, W, BONES)
    log(f'{name}: {len(V)} verts {len(F)} tris')
    return obj


def dispatch(*fns):
    def f(Pt, N, ids):
        out = np.zeros((len(Pt), 3))
        for i, fn in enumerate(fns):
            m = ids == i
            if m.any():
                out[m] = fn(Pt[m], N[m])
        return out
    return f


# ---------------------------------------------------------------- weights
def head_w(V, F):
    Wb = WT.blend_by(V, ['spine_03', 'neck_01', 'Head'], lambda v: v[:, 1], [1.47, 1.535, 1.605])
    W = np.where((V[:, 1] > 1.6)[:, None], WT.rigid(V, 'Head'), Wb)
    return WT.limit(WT.smooth(W, F, 4))


rigid_head = lambda V, F: WT.rigid(V, 'Head')
NOT_LIMBS = [b for b in BONES if not any(k in b for k in ['index', 'middle', 'ring', 'pinky', 'thumb', 'hand', 'thigh', 'calf', 'foot', 'ball', 'Head'])]

# ---------------------------------------------------------------- head + ears
# MakeHuman-generated head + neck (ears included), face photo warped onto its landmarks
_mh = np.load(os.path.join(CACHE, 'mh_head.npz'))
head_sub = sub('head', *load('head'), 30000, head_w, subdiv=1)      # MakeHuman's 'smooth' (4.4k-vertex head)
_mode = json.load(open(os.path.join(HERE, 'face_tweaks.json'))).get('texture_mode', 'photo')
head = assemble('Head', [head_sub], dispatch(PT.paint_head_clean if _mode == 'clean' else PT.paint_head),
                size=2048, rough=0.62, normal=0.35, detail_sigma=2.5, color=(0.72, 0.50, 0.38),
                ao=dict(rays=48, max_dist=0.02, strength=0.35),
                orm_paint=dispatch(PT.head_roughness) if _mode == 'clean' else None)

# MakeHuman high-poly eyeballs with MakeHuman's brown iris texture
_mh = np.load(os.path.join(CACHE, 'mh_head.npz'))
eyes = make_mesh('Eyes', _mh['EV'], _mh['EF'])
import blender_util as BU
set_uv(eyes, _mh['EUV'][BU.LAST_KEPT].reshape(-1, 2))
skin(eyes, arm, WT.rigid(_mh['EV'], 'Head'), BONES)
if PREVIEW:
    eyes.data.materials.append(flat_material('M_Eyes', (0.9, 0.9, 0.9), 0.15))
else:
    # MakeHuman's brown iris is very red under ACES: shift to a darker, less saturated brown
    from PIL import Image as _I
    e = np.asarray(_I.open(os.path.join(P.ROOT, 'build/makehuman/eyes/brown_eye.png')).convert('RGB')).astype(np.float32) / 255
    # bigger iris (less white showing, like the client's): radial remap around both iris centres
    _yy, _xx = np.mgrid[0:e.shape[0], 0:e.shape[1]].astype(np.float32)
    _sx, _sy = _xx.copy(), _yy.copy()
    for _cx, _cy in ((721.0, 299.7), (295.0, 721.7)):
        _r = np.hypot(_xx - _cx, _yy - _cy); R0, k = 118.0, 0.95
        f = np.where(_r < R0 * k, 1 / k, 1 / k + (1 - 1 / k) * np.clip((_r - R0 * k) / (R0 * 0.5), 0, 1))
        m = _r < R0 * (k + 0.5)
        _sx[m] = _cx + (_xx[m] - _cx) * f[m]; _sy[m] = _cy + (_yy[m] - _cy) * f[m]
    e = e[np.clip(_sy.round().astype(int), 0, e.shape[0] - 1), np.clip(_sx.round().astype(int), 0, e.shape[1] - 1)]
    lum = e.mean(2, keepdims=True)
    sat = e.max(2, keepdims=True) - e.min(2, keepdims=True)
    iris = np.clip((sat - 0.12) / 0.15, 0, 1)
    tint = np.array([0.20, 0.17, 0.15])
    e2 = e * (1 - iris) + (lum * 1.25 * tint / tint.mean() * 0.8) * iris          # near black-brown iris
    white = np.clip((lum - 0.55) / 0.2, 0, 1) * (1 - iris)
    e2 = e2 * (1 - white) + e2 * np.array([0.55, 0.44, 0.44]) * white            # not paper-white sclera
    from texlib import save_rgb as _sv
    _sv(os.path.join(TEXDIR, 'Eyes_albedo.jpg'), np.clip(e2, 0, 1), quality=94)
    eyes.data.materials.append(image_material('M_Eyes', os.path.join(TEXDIR, 'Eyes_albedo.jpg'), None, 0.22))
log('Eyes', len(_mh['EV']))

def build_hair():
    """Hair cap + bun with direct UVs into the (upscaled) hair and bun swatches:
    strands flow from the hairline into the bun without any re-baking."""
    subs = [sub('hair', *load('hair'), 14000, rigid_head), sub('bun', *load('bun'), 2400, rigid_head)]
    V = np.concatenate([subs[0]['V'], subs[1]['V']])
    F = np.concatenate([subs[0]['F'], subs[1]['F'] + len(subs[0]['V'])])
    W = np.concatenate([subs[0]['W'], subs[1]['W']])
    which = np.r_[np.zeros(len(subs[0]['F']), int), np.ones(len(subs[1]['F']), int)]
    obj = make_mesh('Hair', V, F)
    import blender_util as BU
    F = F[BU.LAST_KEPT]; which = which[BU.LAST_KEPT]
    UV = np.zeros((len(F), 3, 2))
    for k in range(3):
        for wi, name in enumerate(['hair_top', 'bun']):
            m = which == wi
            uv, ang = PT.hair_uv(V[F[m, k]], name)
            UV[m, k] = uv
    # faces straddling the polar seam (behind the gather point) -> collapse to one side
    for wi in (0, 1):
        m = which == wi
        du = UV[m][:, :, 0].max(1) - UV[m][:, :, 0].min(1)
        bad = np.nonzero(m)[0][du > 0.08]
        UV[bad] = UV[bad][:, :1]
    set_uv(obj, UV.reshape(-1, 2))
    if PREVIEW:
        mat = flat_material('M_Hair', (0.10, 0.065, 0.045), 0.55)
    else:
        path = os.path.join(TEXDIR, 'Hair_albedo.jpg')
        rgb = PT.hair_atlas(path)
        from texlib import normal_from_albedo, save_rgb as _sv
        _sv(os.path.join(TEXDIR, 'Hair_normal.jpg'), normal_from_albedo(rgb, strength=1.3, detail_sigma=2.0), quality=93)
        mat = image_material('M_Hair', path, os.path.join(TEXDIR, 'Hair_normal.jpg'), 0.55)
    mat.use_backface_culling = True
    obj.data.materials.append(mat)
    skin(obj, arm, W, BONES)
    log('Hair', len(V))


def clip_mesh(V, F, N, g):
    """Keep the part of a triangle mesh where the scalar field g >= 0, cutting
    triangles exactly along the g = 0 contour (shared edge points, winding
    preserved). Returns (V, F, N, g) of the clipped mesh."""
    inside = g[F] >= 0
    n_in = inside.sum(1)
    keep = F[n_in == 3]
    nv = len(V)
    new_tris, edges = [], []

    def epoint(i, j):
        edges.append(np.stack([i, j], 1))
        return len(edges) - 1
    # one vertex inside: (a, E(ab), E(ac))
    m1 = n_in == 1
    if m1.any():
        k = np.argmax(inside[m1], 1)
        T = F[m1]
        a_, b_, c_ = T[np.arange(len(T)), k], T[np.arange(len(T)), (k + 1) % 3], T[np.arange(len(T)), (k + 2) % 3]
        new_tris.append(('one', a_, epoint(a_, b_), epoint(a_, c_)))
    # two inside: c outside -> (a, b, E(bc)), (a, E(bc), E(ca))
    m2 = n_in == 2
    if m2.any():
        k = np.argmin(inside[m2], 1)
        T = F[m2]
        c_, a_, b_ = T[np.arange(len(T)), k], T[np.arange(len(T)), (k + 1) % 3], T[np.arange(len(T)), (k + 2) % 3]
        new_tris.append(('two', a_, b_, epoint(b_, c_), epoint(c_, a_)))
    if not edges:
        used = np.unique(keep); rm = -np.ones(nv, int); rm[used] = np.arange(len(used))
        return V[used], rm[keep], N[used], g[used]
    allE = np.concatenate(edges)
    key = np.minimum(allE[:, 0], allE[:, 1]) * nv + np.maximum(allE[:, 0], allE[:, 1])
    ukey, inv = np.unique(key, return_inverse=True)
    i0, j0 = ukey // nv, ukey % nv
    t = g[i0] / (g[i0] - g[j0])
    EV = V[i0] + (V[j0] - V[i0]) * t[:, None]
    EN = N[i0] + (N[j0] - N[i0]) * t[:, None]
    EN /= np.linalg.norm(EN, axis=1, keepdims=True)
    offs = np.cumsum([0] + [len(e) for e in edges])
    eid = lambda block: nv + inv[offs[block]:offs[block + 1]]
    tris = [keep]
    blk = 0
    for item in new_tris:
        if item[0] == 'one':
            _, a_, e1, e2 = item
            tris.append(np.stack([a_, eid(e1), eid(e2)], 1))
        else:
            _, a_, b_, e1, e2 = item
            tris.append(np.stack([a_, b_, eid(e1)], 1)); tris.append(np.stack([a_, eid(e1), eid(e2)], 1))
    FF = np.concatenate(tris)
    VV = np.concatenate([V, EV]); NN = np.concatenate([N, EN]); GG = np.concatenate([g, np.zeros(len(EV))])
    used = np.unique(FF); rm = -np.ones(len(VV), int); rm[used] = np.arange(len(used))
    return VV[used], rm[FF], NN[used], GG[used]


def hair_cap():
    """Hair as one shell grown from the scalp: the long slicked-back top
    (lifted toward the front) continuing into the thicker clipper-cut sides,
    temples and sideburns. The shell is cut exactly along the hairline /
    clipped-hair contour and is flush with the scalp at that edge, so the
    border is smooth (no triangle saw-teeth, no visor). Ears stay bare."""
    V, F = head_sub['V'], head_sub['F']
    N = np_vertex_normals(V, F)
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
    for _ in range(4):                                   # smooth normals: no folds where the shell is thick
        acc = N.copy(); cnt = np.ones(len(V))
        np.add.at(acc, E[:, 0], N[E[:, 1]]); np.add.at(cnt, E[:, 0], 1)
        np.add.at(acc, E[:, 1], N[E[:, 0]]); np.add.at(cnt, E[:, 1], 1)
        N = acc / cnt[:, None]
    N /= np.linalg.norm(N, axis=1, keepdims=True)
    def field(X):
        above = X[:, 1] - P.hair_line(X)
        cd = PT.clipped_density(X)
        return np.maximum(above / 0.004, (cd - 0.50) / 0.30), above, cd
    g, _, _ = field(V)
    V2, F2, N2, g2 = clip_mesh(V, F, N, g)
    _, above2, cd2 = field(V2)
    front = np.clip((V2[:, 2] - P.HEAD_CZ) / 0.09, 0, 1)
    dome = 1 - 0.5 * PT.smoothstep(0.025, 0.045, np.abs(V2[:, 0]))              # rounded dome, no square shoulders
    lift = (0.003 + 0.005 * front) * dome * PT.smoothstep(0.0, 0.016, above2)
    side_t = 0.0045 * PT.smoothstep(0.35, 0.8, cd2)                       # thick, fuzzy clipped sides
    top_t = (0.0024 + lift) * PT.smoothstep(0.0, 0.004, above2)
    t = np.maximum(side_t, top_t) * PT.smoothstep(0.0, 1.0, np.clip(g2, 0, 1))   # flush at the cut edge
    return V2 + N2 * (t + 0.0003)[:, None], F2


capV, capF = hair_cap()
hair = assemble('Hair', [dict(name='cap', V=capV, F=capF, W=WT.rigid(capV, 'Head')), sub('bun', *load('bun'), 2400, rigid_head)],
                dispatch(PT.paint_hair, PT.paint_bun), size=2048, rough=0.58, normal=0.45, orm_paint=dispatch(PT.hair_roughness, PT.bun_roughness), detail_sigma=2.0, color=(0.15, 0.13, 0.11),
                ao=dict(rays=24, max_dist=0.02, strength=0.4))

# ---------------------------------------------------------------- clothing
tee_sub = sub('tee', *load('tee'), 7000)
PT.set_tee_ref(tee_sub['V'])
assemble('Tee', [tee_sub], dispatch(PT.paint_tee), size=1024, rough=0.88, normal=1.4, double_sided=True, color=(0.02, 0.02, 0.022))

jeans_sub = sub('jeans', *load('jeans'), 9000, lambda V, F: WT.limit(WT.smooth(WT.transfer(V, allowed=['pelvis', 'spine_01', 'thigh_l', 'thigh_r', 'calf_l', 'calf_r', 'foot_l', 'foot_r']), F, 8)))
PT.set_jeans_ref(jeans_sub['V'])
assemble('Jeans', [jeans_sub], dispatch(PT.paint_jeans), size=1024, rough=0.82, normal=1.6, double_sided=True, color=(0.06, 0.065, 0.08))

sV, sF = load('shoe_l')
shoe_w = lambda s: (lambda V, F: WT.limit(WT.smooth(WT.transfer(V, allowed=[f'calf_{s}', f'foot_{s}', f'ball_{s}']), F, 4)))
assemble('Shoes', [sub('shoe_l', sV, sF, 3500, shoe_w('l')), sub('shoe_r', *mirror(sV, sF), 3500, shoe_w('r'))],
         dispatch(lambda p, n: PT.paint_shoe(p, n, 'L'), lambda p, n: PT.paint_shoe(p, n, 'R')),
         size=1024, rough=0.72, normal=1.5, color=(0.025, 0.025, 0.027))

aV, aF = load('hand_l')
arm_w = lambda s: (lambda V, F: WT.limit(WT.smooth(WT.hand_weights(V, s), F, 2)))
assemble('Arms', [sub('arm_l', aV, aF, 7000, arm_w('l')), sub('arm_r', *mirror(aV, aF), 7000, arm_w('r'))],
         dispatch(lambda p, n: PT.paint_arm(p, n, 'L'), lambda p, n: PT.paint_arm(p, n, 'R')),
         size=1024, rough=0.55, normal=0.8, detail_sigma=2.5, color=(0.72, 0.50, 0.38),
         ao=dict(rays=24, max_dist=0.025, strength=0.7))

coat_w = lambda V, F: WT.limit(WT.smooth(WT.transfer(V, allowed=NOT_LIMBS), F, 10))
collar_w = lambda V, F: WT.limit(WT.smooth(WT.transfer(V, allowed=['spine_03', 'neck_01', 'clavicle_l', 'clavicle_r']), F, 6))
cuV, cuF = load('cuff_l')
yoke_sub = sub('coat', *load('coat'), 16000, coat_w)
assemble('Coat', [yoke_sub,
                  sub('collar', *load('collar'), 1800, collar_w),
                  sub('cuff_l', cuV, cuF, 1200, lambda V, F: WT.rigid(V, 'upperarm_l')),
                  sub('cuff_r', *mirror(cuV, cuF), 1200, lambda V, F: WT.rigid(V, 'upperarm_r'))],
         dispatch(PT.paint_coat, PT.paint_collar, lambda p, n: PT.paint_cuff(p, n, 'L'), lambda p, n: PT.paint_cuff(p, n, 'R')),
         size=2048, rough=0.86, normal=1.2, double_sided=True, color=(0.85, 0.86, 0.87))

# ---------------------------------------------------------------- free-hanging coat body (simulated at runtime)
# Row 0 of the grid is the seam with the rigid yoke: it follows the chest,
# shoulder blades and - at the armholes - the upper arms. The rest of the
# coat is driven only by the cloth solver. Skin weights below are just a
# fallback for engines without the cloth runtime.
skV, skF = load('skirt')
meta = json.load(open(os.path.join(CACHE, 'skirt.json')))
rows, cols = meta['rows'], meta['cols']


def cloth_w(V, F):
    from scipy.spatial import cKDTree
    n = len(V)
    row = np.arange(n) // cols
    # the seam row must move exactly like the yoke edge it is sewn to (no tearing),
    # so copy the yoke's own weights: shoulders/shoulder blades, and the upper
    # arms around the armholes
    yV, yW = yoke_sub['V'], yoke_sub['W']
    d, idx = cKDTree(yV).query(V, k=6)
    w = 1.0 / np.maximum(d, 1e-4) ** 2
    W = (yW[idx] * w[..., None]).sum(1) / w.sum(1, keepdims=True)
    W = WT.normalize(W)
    # lower rows (fallback only) fade toward the spine/pelvis
    low = WT.transfer(V, allowed=['spine_01', 'spine_02', 'pelvis', 'thigh_l', 'thigh_r'])
    s = np.clip(row / (rows - 1) * 1.6, 0, 1)
    W = WT.normalize(W * (1 - s[:, None]) + low * s[:, None])
    return WT.limit(W)


cloth = make_mesh('CoatCloth', skV, skF)
import blender_util as BU
skF = skF[BU.LAST_KEPT]
idx = np.arange(len(skV))
grid_uv = np.c_[(idx % cols) / (cols - 1), 1 - (idx // cols) / (rows - 1)]
set_uv(cloth, grid_uv[np.asarray(skF).ravel()], 'UVMap')
set_uv(cloth, np.c_[idx % cols, idx // cols][np.asarray(skF).ravel()].astype(float), 'ClothGrid')
skin(cloth, arm, cloth_w(skV, skF), BONES)
if PREVIEW:
    m = flat_material('M_CoatCloth', (0.85, 0.86, 0.87), 0.86); m.use_backface_culling = False
elif os.environ.get('REBAKE') and 'CoatCloth' not in os.environ['REBAKE'] and os.path.exists(os.path.join(TEXDIR, 'CoatCloth_albedo.jpg')):
    m = image_material('M_CoatCloth', os.path.join(TEXDIR, 'CoatCloth_albedo.jpg'), os.path.join(TEXDIR, 'CoatCloth_normal.jpg'), 0.86, double_sided=True)
else:
    N = np_vertex_normals(skV, skF)
    rgb, mask = bake(skV, N, skF, grid_uv[skF], PT.paint_coat, 2048)
    maps = write_maps(os.path.join(TEXDIR, 'CoatCloth'), rgb, mask, normal_strength=1.2)
    m = image_material('M_CoatCloth', maps['albedo'], maps.get('normal'), 0.86, double_sided=True)
cloth.data.materials.append(m)
cloth['cloth'] = json.dumps({k: meta[k] for k in ['rows', 'cols', 'ventCol', 'ventRow']})
log('CoatCloth', rows, cols)

# ---------------------------------------------------------------- export
bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_animations=True, export_animation_mode='ACTIONS',
                          export_skins=True, export_extras=True, export_yup=True, export_apply=False,
                          export_force_sampling=True, export_def_bones=False, export_image_format='AUTO',
                          export_texcoords=True, export_normals=True, export_morph=False)
log('exported', OUT)
