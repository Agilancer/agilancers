"""Side-by-side likeness sheet: the client's video frames vs. the model at
matching angles. usage: face_compare.py <out.png>"""
import os, sys, subprocess, json
import numpy as np
from PIL import Image, ImageDraw
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCR = sys.argv[2] if len(sys.argv) > 2 else '/tmp'
FRAMES = os.environ.get('HEADVID', '')
# (video frame, camera yaw in degrees: + = camera toward the character's left side)
CROP = (180, 1400, 1980, 3476)          # client frame window shown in the sheet
CD = float(os.environ.get('CMP_DIST', '0.46'))     # ~0.38 m eye-to-phone, front camera at arm's length
CLIENT_EYES = {'f_015.jpg': (804, 2252)}            # pupil midpoint in the frame     # camera distance from the head centre
DROP = float(os.environ.get('CMP_DROP', '0.07'))   # phone below eye level
PAIRS = [('f_001.jpg', 68), ('f_007.jpg', 38), ('f_015.jpg', 0), ('f_023.jpg', -32), ('f_031.jpg', -78)]


def main(out):
    views = [f'cmp{yaw}' for f, yaw in PAIRS]
    cam = f'fov=48&cd={CD}&drop={DROP}&hide=Arms'
    url = f'dev/preview.html?glb=../assets/models/agilancer_hero.glb&{cam}&views=' + ','.join(views)
    tmp = os.path.join(SCR, 'cmp.png')
    subprocess.run(['node', 'tools/shot.cjs', url, tmp, '520', '600', '--noshot=1'], cwd=ROOT, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    # the large frontal pair is a 2x render of the same camera, cropped like the client frame (no stretching)
    big = os.path.join(SCR, 'cmpbig.png')
    subprocess.run(['node', 'tools/shot.cjs', f'dev/preview.html?glb=../assets/models/agilancer_hero.glb&{cam}&views=cmp0',
                    big, '1040', '1200', '--noshot=1'], cwd=ROOT, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    cols = []
    for (f, yaw), v in zip(PAIRS, views):
        ref = Image.open(os.path.join(FRAMES, f)).convert('RGB').crop(CROP).resize((520, 600))
        ren = Image.open(tmp.replace('.png', f'_{v}.png')).convert('RGB')
        col = Image.new('RGB', (520, 1200)); col.paste(ref, (0, 0)); col.paste(ren, (0, 600))
        cols.append(col)
    sheet = Image.new('RGB', (520 * len(cols), 1240), (20, 20, 24))
    for i, c in enumerate(cols):
        sheet.paste(c, (i * 520, 40))
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), 'TOP ROW: the client (video frames)    BOTTOM ROW: the model at matching angles', fill=(255, 255, 255))
    sheet.resize((sheet.width * 2 // 3, sheet.height * 2 // 3)).save(out, quality=92)
    # large frontal pair for detail: same crop window on both sides
    CL = (330, 1700, 1290, 2980)
    ref = Image.open(os.path.join(FRAMES, 'f_015.jpg')).convert('RGB').crop(CL).resize((600, 800))
    # crop the model render around its projected eyes so they land where the client's pupils are
    lm = json.load(open(os.path.join(ROOT, 'build/cache/mh_landmarks.json')))
    eye = 0.5 * (np.array(lm['eye_l']) + np.array(lm['eye_r']))
    c = np.array([0, 1.675, 0.035]); cp = np.array([0, 1.695 - DROP, 0.035 + CD])
    fwd = (c - cp) / np.linalg.norm(c - cp); rt = np.cross(fwd, [0, 1, 0]); rt /= np.linalg.norm(rt); up = np.cross(rt, fwd)
    v = eye - cp; f = 600 / np.tan(np.radians(24))           # 2x render: 1200 px tall, 48 deg
    ex, ey_ = 520 + f * (v @ rt) / (v @ fwd), 600 - f * (v @ up) / (v @ fwd)
    k = 2 * 520 / (CROP[2] - CROP[0])
    px, py = CLIENT_EYES['f_015.jpg']
    box = (ex - (px - CL[0]) * k, ey_ - (py - CL[1]) * k)
    box = tuple(int(round(b)) for b in (box[0], box[1], box[0] + (CL[2] - CL[0]) * k, box[1] + (CL[3] - CL[1]) * k))
    ren = Image.open(big.replace('.png', '_cmp0.png')).convert('RGB').crop(box).resize((600, 800))
    pair = Image.new('RGB', (1200, 800)); pair.paste(ref, (0, 0)); pair.paste(ren, (600, 0))
    pair.save(out.replace('.jpg', '_front.jpg'), quality=92)


if __name__ == '__main__':
    main(sys.argv[1])
