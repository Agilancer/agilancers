"""Export the 10 lance + 10 shield designs to assets/models/agilancer_arms.glb.
Each design gets its own material using its sheet panel as the texture (with
metal/roughness derived from colour)."""
import os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..')); sys.path.insert(0, os.path.join(HERE, '..', 'hero'))
import bpy
from PIL import Image
from scipy import ndimage
from arms_geom import lance_mesh, shield_mesh, ROOT, PIECES
from weapon_layout import VARIANTS, VARIANT_LABELS
from blender_util import make_mesh, set_uv, image_material, decimate, triangulate, mesh_arrays
from texlib import normal_from_albedo, save_rgb

OUT = os.path.join(ROOT, 'assets/models/agilancer_arms.glb')
TEX = os.path.join(ROOT, 'build', 'textures', 'arms'); os.makedirs(TEX, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
t0 = time.time()


def orm_from_albedo(rgb):
    """glTF ORM: R occlusion=1, G roughness, B metalness - guessed from colour:
    bright/low-saturation or golden pixels read as polished metal."""
    mx, mn = rgb.max(2), rgb.min(2)
    sat = (mx - mn) / np.maximum(mx, 1e-3)
    lum = rgb.mean(2)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    gold = np.clip(((r > g) & (g > b) & (sat > 0.35) & (lum > 0.35)).astype(float), 0, 1)
    silver = np.clip((sat < 0.12) * np.clip((lum - 0.45) / 0.25, 0, 1), 0, 1)
    metal = ndimage.gaussian_filter(np.maximum(gold, silver * 0.85), 1.0)
    rough = np.clip(0.75 - 0.45 * metal + 0.15 * (1 - lum), 0.25, 0.95)
    return np.dstack([np.ones_like(rough), rough, metal])


def material_for(kind, key):
    src = os.path.join(PIECES, f'{kind}_{key}.png')
    rgb = np.asarray(Image.open(src).convert('RGB')).astype(np.float32) / 255
    base = os.path.join(TEX, f'{kind}_{key}')
    save_rgb(base + '_albedo.jpg', rgb, quality=92)
    save_rgb(base + '_normal.jpg', normal_from_albedo(rgb, strength=2.2, detail_sigma=3.0), quality=92)
    save_rgb(base + '_orm.png', orm_from_albedo(rgb))
    return image_material(f'M_{kind.title()}_{key}', base + '_albedo.jpg', base + '_normal.jpg', 0.6, 0.0, orm_path=base + '_orm.png', normal_strength=1.0)


for key in VARIANTS:
    for kind, fn, target in (('lance', lance_mesh, 9000), ('shield', shield_mesh, 9000)):
        V, F, UV = fn(key)
        name = f'{kind.title()}_{key}'
        obj = make_mesh(name, V, F)
        import blender_util as BU
        set_uv(obj, UV[BU.LAST_KEPT].reshape(-1, 2))
        decimate(obj, target)
        triangulate(obj)
        obj.data.materials.append(material_for(kind, key))
        print(name, len(obj.data.polygons), f'{time.time()-t0:.0f}s', flush=True)

bpy.ops.export_scene.gltf(filepath=OUT, export_format='GLB', export_animations=False, export_skins=False,
                          export_yup=True, export_texcoords=True, export_normals=True, export_image_format='AUTO')
print('exported', OUT, f'{time.time()-t0:.0f}s')
