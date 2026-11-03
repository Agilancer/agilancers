"""Nose close-ups: client frames vs the model at matching angles, each cropped
around the nose tip (model tip projected through the comparison camera).
usage: nose_compare.py <iterdir>   (needs the cmp_cmp*.png renders from face_compare.py)"""
import os, sys, json
import numpy as np
from PIL import Image
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import face_compare as FC
# client nose-tip pixel in each frame (measured)
TIPS = {'f_001.jpg': (430, 2360), 'f_007.jpg': (700, 2500), 'f_015.jpg': (804, 2490), 'f_031.jpg': (1130, 2490)}
# client chin (pogonion region, incl. beard) in each frame
CHINS = {'f_001.jpg': (450, 2650), 'f_007.jpg': (1000, 2930), 'f_015.jpg': (804, 2960), 'f_023.jpg': (800, 2930), 'f_031.jpg': (1110, 2740)}


def project(p, yaw, W=520, H=600):
    c = np.array([0, 1.675, 0.035]); y = np.radians(yaw)
    cp = c + np.array([np.sin(y) * FC.CD, 0.02 - FC.DROP, np.cos(y) * FC.CD])
    fwd = (c - cp) / np.linalg.norm(c - cp); rt = np.cross(fwd, [0, 1, 0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fwd)
    v = np.asarray(p) - cp; f = (H / 2) / np.tan(np.radians(24))
    return W / 2 + f * (v @ rt) / (v @ fwd), H / 2 - f * (v @ up) / (v @ fwd)


def main(it, feature='nose'):
    lm = json.load(open(os.path.join(ROOT, 'build/cache/mh_landmarks.json')))
    k = 520 / (FC.CROP[2] - FC.CROP[0]); half = 70
    tiles = []
    pts = TIPS if feature == 'nose' else CHINS
    key = 'tip' if feature == 'nose' else 'pogonion'
    for f, yaw in FC.PAIRS:
        if f not in pts:
            continue
        ren = Image.open(os.path.join(it, f'cmp_cmp{yaw}.png')).convert('RGB')
        mx, my = project(lm[key], yaw)
        cx, cy = pts[f]
        ref = Image.open(os.path.join(FC.FRAMES, f)).convert('RGB')
        hr = half / k
        a = ref.crop((int(cx - 1.3 * hr), int(cy - 1.6 * hr), int(cx + 1.3 * hr), int(cy + 1.0 * hr))).resize((364, 364))
        b = ren.crop((int(mx - 1.3 * half), int(my - 1.6 * half), int(mx + 1.3 * half), int(my + 1.0 * half))).resize((364, 364))
        tiles.append((a, b))
    out = Image.new('RGB', (364 * len(tiles), 728))
    for i, (a, b) in enumerate(tiles):
        out.paste(a, (i * 364, 0)); out.paste(b, (i * 364, 364))
    out.save(os.path.join(it, f'{feature}_compare.jpg'), quality=92)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else 'nose')
