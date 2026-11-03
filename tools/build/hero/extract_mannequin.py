"""Dump the UAL2 mannequin mesh, skin weights and skeleton (bind pose) to
build/mannequin.npz (glTF axes) so the numpy modelling stages can use it."""
import bpy, sys, os, numpy as np
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=os.path.join(ROOT, 'assets/models/UAL2_Standard.glb'))
man = bpy.data.objects['Mannequin']
arm = bpy.data.objects['Armature']
me = man.data
me.calc_loop_triangles()

def g(v):  # blender -> glTF axes
    return (v[0], v[2], -v[1])

V = np.array([g(man.matrix_world @ v.co) for v in me.vertices])
T = np.array([t.vertices[:] for t in me.loop_triangles])
TM = np.array([t.material_index for t in me.loop_triangles])
bones = [b.name for b in arm.data.bones]
gi = {grp.index: grp.name for grp in man.vertex_groups}
W = np.zeros((len(V), len(bones)), np.float32)
for v in me.vertices:
    for gg in v.groups:
        W[v.index, bones.index(gi[gg.group])] = gg.weight
heads = np.array([g(arm.matrix_world @ b.head_local) for b in arm.data.bones])
tails = np.array([g(arm.matrix_world @ b.tail_local) for b in arm.data.bones])
parents = np.array([bones.index(b.parent.name) if b.parent else -1 for b in arm.data.bones])
os.makedirs(os.path.join(ROOT, 'build'), exist_ok=True)
np.savez(os.path.join(ROOT, 'build/mannequin.npz'), V=V, T=T, TM=TM, W=W, bones=np.array(bones), heads=heads, tails=tails, parents=parents)
print('verts', len(V), 'tris', len(T), 'bones', len(bones))
for i, b in enumerate(bones):
    if 'leaf' in b: continue
    print(b, np.round(heads[i], 4), np.round(tails[i], 4))
