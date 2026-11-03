"""Stage 1 of the hero build: slice the provided texture sheets into clean
pieces (backdrop removed, colours edge-padded) that the Blender build script
maps onto the new meshes.

    python tools/build/textures.py          -> build/pieces/*.png + pieces.json
"""
import json, os, sys
import numpy as np
from scipy import ndimage
sys.path.insert(0, os.path.dirname(__file__))
from texlib import load_rgb, cut_piece, dilate_colors, save_rgba, save_rgb
from weapon_layout import *

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
OUT = os.path.join(ROOT, 'build', 'pieces')
os.makedirs(OUT, exist_ok=True)

# Outfit sheet piece boxes (x0, y0, x1, y1), found by segmenting the sheet.
OUTFIT = {
    'coat_front_r': (5, 5, 183, 443),     # wearer's right front panel (buttonholes)
    'coat_front_l': (214, 4, 389, 443),   # wearer's left front panel (chest pocket, buttons)
    'coat_back':    (415, 4, 787, 441),
    'sleeve_a':     (803, 18, 1011, 302),
    'sleeve_b':     (1027, 18, 1239, 302),
    'cuff_a':       (813, 344, 1007, 405),
    'cuff_b':       (1035, 344, 1230, 408),
    'tee_front':    (60, 450, 370, 778),
    'tee_back':     (399, 453, 703, 777),
    'forearm_a':    (731, 453, 834, 780),
    'forearm_b':    (869, 453, 981, 781),
    'hand_dorsal_l': (1005, 439, 1118, 601),
    'hand_dorsal_r': (1122, 439, 1235, 601),
    'hand_palm_l':  (983, 605, 1106, 780),
    'hand_palm_r':  (1118, 605, 1241, 781),
    'jeans_front':  (10, 788, 243, 1250),
    'jeans_back':   (251, 789, 454, 1250),
    'shoe_side_a':  (465, 788, 714, 897),   # toe points left
    'shoe_side_b':  (466, 911, 715, 1022),  # toe points right
    'sole_side_a':  (464, 1044, 723, 1091),
    'sole_side_b':  (462, 1101, 717, 1146),
    'outsole':      (460, 1153, 725, 1250),
    'head':         (719, 793, 1043, 1248),
    'ear_a':        (1068, 792, 1143, 917),
    'ear_b':        (1160, 792, 1235, 917),
    'hair_top':     (1038, 920, 1251, 1037),
    'stubble_a':    (1043, 1035, 1135, 1108),
    'stubble_b':    (1154, 1036, 1244, 1108),
    'stubble_c':    (1037, 1116, 1164, 1237),
    'bun':          (1156, 1109, 1247, 1203),
}


def outfit():
    sheet = load_rgb(os.path.join(ROOT, 'assets/source/outfit_sheet.png'))
    meta = {}
    for name, box in OUTFIT.items():
        rgba, (ox, oy) = cut_piece(sheet, box, tol=0.07, pad=4)
        rgb = dilate_colors(rgba, iters=24)
        save_rgba(os.path.join(OUT, name + '.png'), np.dstack([rgb, rgba[..., 3]]))
        ys, xs = np.nonzero(rgba[..., 3] > 0.5)
        meta[name] = {'origin': [ox, oy], 'size': [rgba.shape[1], rgba.shape[0]],
                      'bbox': [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]}
    return meta


def panel_mask(crop, tol=0.07, std_thr=0.012, fill=True, inset=0):
    i = inset
    border = np.concatenate([crop[i], crop[-1 - i], crop[:, i], crop[:, -1 - i]])
    bg = np.median(border, axis=0)
    d = np.abs(crop - bg).max(2)
    sat = crop.max(2) - crop.min(2)
    lum = crop.mean(2)
    # the backdrop is flat; objects are textured -> local std is a second cue
    mu = ndimage.uniform_filter(lum, 5)
    std = np.sqrt(np.maximum(ndimage.uniform_filter(lum * lum, 5) - mu * mu, 0))
    fg = ~((d < tol) & (sat < 0.06) & (std < std_thr))
    fg = ndimage.binary_opening(fg, iterations=2)
    if fill:
        fg = ndimage.binary_fill_holes(fg)
    return fg


def smooth_mask(m, close=4, sigma=1.5):
    m = ndimage.binary_closing(np.pad(m, close + 2), iterations=close)[close + 2:-close - 2, close + 2:-close - 2]
    m = ndimage.binary_fill_holes(m)
    return ndimage.gaussian_filter(m.astype(np.float32), sigma) > 0.5


def lances():
    sheet = load_rgb(os.path.join(ROOT, LANCE_SHEET))
    pw, ph = 251, 627
    meta = {}
    for i, key in enumerate(VARIANTS):
        ox, oy = lance_panel_origin(i)
        crop = sheet[oy:oy + ph, ox:ox + pw]
        crop = np.pad(crop, ((0, max(0, ph - crop.shape[0])), (0, max(0, pw - crop.shape[1])), (0, 0)), mode='edge')
        # mask separator lines out by shrinking the panel border region
        fg = panel_mask(crop[4:-4, 4:-4])
        fg = np.pad(fg, 4)
        # blade mask: union of components inside the headA rect
        x0, y0, x1, y1 = LANCE_RECTS['headA']
        sub = panel_mask(crop[y0:y1, x0:x1], tol=0.06, std_thr=0.02, fill=False, inset=5)
        sub[:3] = False; sub[-2:] = False; sub[:, :3] = False; sub[:, -3:] = False   # drop separator lines
        sub = ndimage.binary_opening(sub, iterations=2)
        lab, n = ndimage.label(sub)
        if n > 1:
            sizes = ndimage.sum(sub, lab, range(1, n + 1))
            sub = lab == 1 + int(np.argmax(sizes))
        sub = smooth_mask(sub, close=3, sigma=1.0)
        lab, n = ndimage.label(sub)
        if n > 1:  # keep components of meaningful size
            sizes = ndimage.sum(sub, lab, range(1, n + 1))
            sub = np.isin(lab, 1 + np.nonzero(sizes > 0.02 * sizes.max())[0])
        blade = np.zeros_like(fg)
        blade[y0:y1, x0:x1] = sub
        rgb = dilate_colors(np.dstack([crop, fg.astype(np.float32)]), iters=30)
        save_rgb(os.path.join(OUT, f'lance_{key}.png'), rgb)
        save_rgba(os.path.join(OUT, f'lance_{key}_blademask.png'), np.dstack([blade, blade, blade, blade]).astype(np.float32))
        meta[key] = {'size': [pw, ph]}
    return meta


def shields():
    sheet = load_rgb(os.path.join(ROOT, SHIELD_SHEET))
    pw, ph = 355, 443
    meta = {}
    seeds = {'back': (95, 372), 'rim': (233, 372), 'grip': (299, 372)}
    for i, key in enumerate(VARIANTS):
        ox, oy = shield_panel_origin(i)
        crop = sheet[oy:oy + ph, ox:ox + pw]
        crop = np.pad(crop, ((0, max(0, ph - crop.shape[0])), (0, max(0, pw - crop.shape[1])), (0, 0)), mode='edge')
        fg = np.pad(panel_mask(crop[4:-4, 4:-4]), 4)
        # marker-based watershed on the colour gradient splits pieces that touch
        from skimage.segmentation import watershed
        from skimage.filters import sobel
        # elevation: low inside thick regions, ridges along thin contact seams
        grad = -ndimage.distance_transform_edt(fg)
        markers = np.zeros(fg.shape, np.int32)
        markers[~fg] = 1
        for j, (name, (sx, sy)) in enumerate([('main', (175, 130))] + list(seeds.items())):
            markers[sy - 5:sy + 6, sx - 5:sx + 6] = j + 2
        lab = watershed(grad, markers, mask=None)
        parts = {}
        for j, name in enumerate(['main'] + list(seeds)):
            m = (lab == j + 2) & fg
            m = ndimage.binary_opening(m, iterations=2)
            l2, n2 = ndimage.label(m)
            if n2 > 1:
                sizes = ndimage.sum(m, l2, range(1, n2 + 1))
                m = l2 == 1 + int(np.argmax(sizes))
            parts[name] = smooth_mask(m)
        main = parts.pop('main')
        rgb = dilate_colors(np.dstack([crop, fg.astype(np.float32)]), iters=30)
        save_rgb(os.path.join(OUT, f'shield_{key}.png'), rgb)
        save_rgba(os.path.join(OUT, f'shield_{key}_mask_main.png'), np.dstack([main] * 4).astype(np.float32))
        save_rgba(os.path.join(OUT, f'shield_{key}_mask_back.png'), np.dstack([parts['back']] * 4).astype(np.float32))
        bb = {}
        for name, m in [('main', main)] + list(parts.items()):
            ys, xs = np.nonzero(m)
            if len(xs) == 0:
                print('EMPTY', key, name); continue
            bb[name] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
        meta[key] = {'size': [pw, ph], 'bbox': bb}
    return meta


if __name__ == '__main__':
    meta = {'outfit': outfit(), 'lance': lances(), 'shield': shields()}
    with open(os.path.join(OUT, 'pieces.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('wrote', len(os.listdir(OUT)), 'files')
