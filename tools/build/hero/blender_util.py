"""bpy helpers: numpy <-> Blender meshes (glTF <-> Blender axes), decimation,
UV unwrapping, skinning and simple materials."""
import bpy, bmesh
import numpy as np


def to_blender(V):
    V = np.asarray(V, float)
    return np.c_[V[:, 0], -V[:, 2], V[:, 1]]


def to_gltf(V):
    V = np.asarray(V, float)
    return np.c_[V[:, 0], V[:, 2], -V[:, 1]]


LAST_KEPT = None


def make_mesh(name, V, F, smooth=True):
    """Create a mesh object. Faces with repeated vertex indices are dropped
    here (instead of by Blender's validate(), which would silently shift the
    loop order); LAST_KEPT holds the mask so per-corner data can follow."""
    global LAST_KEPT
    F = np.asarray(F)
    keep = (F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])
    LAST_KEPT = keep
    F = F[keep]
    me = bpy.data.meshes.new(name)
    Vb = to_blender(V)
    me.vertices.add(len(Vb))
    me.vertices.foreach_set('co', Vb.astype(np.float32).ravel())
    nf = len(F)
    me.loops.add(nf * 3)
    me.loops.foreach_set('vertex_index', np.asarray(F, np.int32).ravel())
    me.polygons.add(nf)
    me.polygons.foreach_set('loop_start', np.arange(0, nf * 3, 3, dtype=np.int32))
    me.polygons.foreach_set('loop_total', np.full(nf, 3, np.int32))
    me.update(calc_edges=True)
    if smooth:
        me.polygons.foreach_set('use_smooth', np.ones(len(me.polygons), bool))
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def mesh_arrays(obj):
    me = obj.data
    me.calc_loop_triangles()
    V = np.zeros(len(me.vertices) * 3); me.vertices.foreach_get('co', V)
    V = to_gltf(V.reshape(-1, 3))
    F = np.zeros(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get('vertices', F)
    return V, F.reshape(-1, 3)


def apply_modifiers(obj):
    dg = bpy.context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(dg)
    me = bpy.data.meshes.new_from_object(ev)
    old = obj.data
    obj.modifiers.clear()
    obj.data = me
    bpy.data.meshes.remove(old)


def decimate(obj, target_tris):
    n = len(obj.data.polygons)
    if n <= target_tris:
        return
    m = obj.modifiers.new('dec', 'DECIMATE')
    m.ratio = target_tris / n
    m.use_collapse_triangulate = True
    apply_modifiers(obj)


def triangulate(obj):
    obj.modifiers.new('tri', 'TRIANGULATE')
    apply_modifiers(obj)


def smart_uv(obj, angle=66.0, margin=0.006):
    for o in bpy.context.selected_objects:
        o.select_set(False)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.uv.smart_project(angle_limit=np.radians(angle), island_margin=margin, area_weight=0.0, scale_to_bounds=False)
    bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)


def set_uv(obj, loop_uv, name='UVMap'):
    me = obj.data
    layer = me.uv_layers.get(name) or me.uv_layers.new(name=name)
    layer.data.foreach_set('uv', np.asarray(loop_uv, np.float32).ravel())
    return layer


def get_loop_uv(obj, name=None):
    me = obj.data
    layer = me.uv_layers[name] if name else me.uv_layers.active
    uv = np.zeros(len(me.loops) * 2, np.float32)
    layer.data.foreach_get('uv', uv)
    return uv.reshape(-1, 2)


def loop_tris(obj):
    """Per-triangle loop indices (mesh must be triangulated)."""
    me = obj.data
    me.calc_loop_triangles()
    L = np.zeros(len(me.loop_triangles) * 3, np.int32)
    me.loop_triangles.foreach_get('loops', L)
    return L.reshape(-1, 3)


def vertex_normals(obj):
    me = obj.data
    N = np.zeros(len(me.vertices) * 3)
    me.vertices.foreach_get('normal', N)
    return to_gltf(N.reshape(-1, 3))


def skin(obj, arm, W, bones, threshold=1e-4):
    for b in bones:
        pass
    nz = np.nonzero(W > threshold)
    groups = {}
    for v, j in zip(*nz):
        groups.setdefault(j, ([], []))
        groups[j][0].append(int(v)); groups[j][1].append(float(W[v, j]))
    for j, (vs, ws) in groups.items():
        vg = obj.vertex_groups.new(name=bones[j])
        # group identical weights to reduce calls
        ws = np.round(np.array(ws), 4)
        for w in np.unique(ws):
            idx = [vs[k] for k in np.nonzero(ws == w)[0]]
            vg.add(idx, float(w), 'REPLACE')
    mod = obj.modifiers.new('Armature', 'ARMATURE')
    mod.object = arm
    obj.parent = arm


def flat_material(name, rgb, rough=0.7, metal=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = (*rgb, 1)
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    return m


def image_material(name, albedo_path, normal_path=None, rough=0.7, metal=0.0, orm_path=None, double_sided=False, normal_strength=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    tex = nt.nodes.new('ShaderNodeTexImage')
    tex.image = bpy.data.images.load(albedo_path)
    nt.links.new(tex.outputs['Color'], bsdf.inputs['Base Color'])
    bsdf.inputs['Roughness'].default_value = rough
    bsdf.inputs['Metallic'].default_value = metal
    if normal_path:
        nt_img = nt.nodes.new('ShaderNodeTexImage')
        nt_img.image = bpy.data.images.load(normal_path)
        nt_img.image.colorspace_settings.name = 'Non-Color'
        nm = nt.nodes.new('ShaderNodeNormalMap')
        nm.inputs['Strength'].default_value = normal_strength
        nt.links.new(nt_img.outputs['Color'], nm.inputs['Color'])
        nt.links.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
    if orm_path:
        orm = nt.nodes.new('ShaderNodeTexImage')
        orm.image = bpy.data.images.load(orm_path)
        orm.image.colorspace_settings.name = 'Non-Color'
        sep = nt.nodes.new('ShaderNodeSeparateColor')
        nt.links.new(orm.outputs['Color'], sep.inputs['Color'])
        nt.links.new(sep.outputs['Green'], bsdf.inputs['Roughness'])
        nt.links.new(sep.outputs['Blue'], bsdf.inputs['Metallic'])
    m.use_backface_culling = not double_sided
    return m


def np_vertex_normals(V, F):
    tri = V[F]
    fn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])   # area weighted
    N = np.zeros_like(V)
    for k in range(3):
        np.add.at(N, F[:, k], fn)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
