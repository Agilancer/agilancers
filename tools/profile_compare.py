"""Profile silhouette comparison: the client's right profile (video frame
f_031, whiteboard background) against the model rendered at the same yaw.
Both outlines are aligned on nasion + subnasale (similarity transform, so
scale/pitch differences of the phone camera cancel), then drawn over the model
render with per-height depth differences in mm.
usage: profile_compare.py <iterdir>   (renders cmp-78 at 2x itself)"""
import os, sys, json, subprocess
import numpy as np
from PIL import Image, ImageDraw
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))
import face_compare as FC

YAW = -78


def client_outline(path, y0=1900, y1=2800, x0=800, x1=1500):
    """Front edge of the face per row against the whiteboard: skin/beard (reddish) or hair (dark) above the
    tray; below y 2660 only reddish pixels count (the tray is dark). A running median removes the thin
    whiteboard lines."""
    a = np.asarray(Image.open(path).convert('RGB')).astype(float)
    xs, ys = [], []
    for y in range(y0, y1):
        row = a[y, x0:x1]
        br = row.mean(1)
        skin = (row[:, 0] > row[:, 2] + 22) & (row[:, 0] > row[:, 1] + 8) & (br > 70)
        fg = skin | ((br < 110) & (y < 2660))
        if y >= 2660:
            fg = skin & (row[:, 0] > row[:, 2] + 30) & (br > 105)
        run = np.convolve(fg.astype(int), np.ones(6, int), 'same') >= 6
        idx = np.nonzero(run)[0]
        if len(idx):
            xs.append(x0 + idx.max()); ys.append(y)
    xs = np.array(xs, float); ys = np.array(ys, float)
    k = 10                                                    # running median (no scipy in this python)
    pad = np.r_[np.full(k, xs[0]), xs, np.full(k, xs[-1])]
    xs = np.array([np.median(pad[i:i + 2 * k + 1]) for i in range(len(xs))])
    return np.c_[xs, ys]


def model_outline(img):
    a = np.asarray(img.convert('RGB')).astype(float)
    bg = np.array([20, 20, 24])
    fg = np.abs(a - bg).sum(2) > 40
    pts = []
    for y in range(a.shape[0]):
        idx = np.nonzero(fg[y])[0]
        if len(idx):
            pts.append((idx.max(), y))
    return np.array(pts, float)


def smooth(P, k=7):
    if len(P) < k:
        return P
    x = np.convolve(P[:, 0], np.ones(k) / k, 'same')
    x[:k] = P[:k, 0]; x[-k:] = P[-k:, 0]
    return np.c_[x, P[:, 1]]


def key_points(P, y_nasion, y_sub, y_tip):
    """nasion: most posterior point between brow and tip; subnasale: most posterior between tip and lip.
    The y_* are rough guesses (row numbers)."""
    def seg(ya, yb):
        return P[(P[:, 1] >= ya) & (P[:, 1] <= yb)]
    tip = seg(y_tip - 60, y_tip + 60); tip = tip[np.argmax(tip[:, 0])]
    nas = seg(y_nasion - 60, tip[1] - 40); nas = nas[np.argmin(nas[:, 0])]
    sub = seg(tip[1] + 10, y_sub + 50); sub = sub[np.argmin(sub[:, 0])]
    return nas, tip, sub


def main(it):
    big = os.path.join(it, 'prof.png')
    CDP = 0.62                                               # a bit further than the likeness sheet: the whole profile fits
    cam = f'fov=48&cd={CDP}&drop={FC.DROP}&hide=Arms'
    subprocess.run(['node', 'tools/shot.cjs', f'dev/preview.html?glb=../assets/models/agilancer_hero.glb&{cam}&views=cmp{YAW}',
                    big, '1040', '1200', '--noshot=1'], cwd=ROOT, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    ren = Image.open(big.replace('.png', f'_cmp{YAW}.png')).convert('RGB')
    M = smooth(model_outline(ren))
    C = smooth(client_outline(os.path.join(FC.FRAMES, 'f_031.jpg')))
    # rough rows: model from the projected landmarks
    lm = json.load(open(os.path.join(ROOT, 'build/cache/mh_landmarks.json')))
    sys.path.insert(0, os.path.dirname(__file__))
    from nose_compare import project
    import nose_compare as NC
    _cd = FC.CD; FC.CD = CDP
    prj = lambda p: np.array(project(p, YAW, 1040, 1200))
    m_nas, m_tip, m_sub = key_points(M, prj(lm['glabella'])[1] + 30, prj(lm['subnasale'])[1], prj(lm['tip'])[1])
    c_nas, c_tip, c_sub = key_points(C, 2320, 2553, 2487)
    # similarity transform client -> model from (nasion, subnasale)
    ca, cb = c_nas, c_sub; ma, mb = m_nas, m_sub
    vc, vm = cb - ca, mb - ma
    s = np.linalg.norm(vm) / np.linalg.norm(vc)
    ang = np.arctan2(vm[1], vm[0]) - np.arctan2(vc[1], vc[0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    T = lambda P: (R @ (P - ca).T).T * s + ma
    C2 = T(C)
    # mm per render pixel at the face: model nasion->subnasale length in 3D
    L3 = abs(lm['glabella'][1] - lm['subnasale'][1])            # glabella is close to the nasion height
    mm_px = (np.linalg.norm(np.array(lm['glabella']) - np.array(lm['subnasale'])) * 1000) / np.linalg.norm(vm)
    # per-row horizontal differences (client - model, + = client sticks out further)
    out = ren.copy(); d = ImageDraw.Draw(out)
    d.line([tuple(p) for p in M], fill=(80, 200, 255), width=2)
    d.line([tuple(p) for p in C2], fill=(255, 170, 40), width=2)
    for p in (m_nas, m_sub, m_tip):
        d.ellipse([p[0] - 4, p[1] - 4, p[0] + 4, p[1] + 4], outline=(80, 200, 255))
    report = []
    # feature rows from the model's own 3D midline (projected), forehead heights above the eye line
    d3 = np.load(os.path.join(ROOT, 'build/cache/mh_head.npz')); V3 = d3['V']
    midl = V3[(np.abs(V3[:, 0]) < 0.0025) & (V3[:, 2] > 0.06)]
    ey = lm['eye_y']
    def front_at(y):
        q = midl[np.abs(midl[:, 1] - y) < 0.0015]
        return q[np.argmax(q[:, 2])] if len(q) else None
    feats = [('forehead +60', ey + 0.060), ('forehead +40', ey + 0.040), ('brow +20', ey + 0.020), ('nasion', None),
             ('dorsum mid', None), ('tip', None), ('subnasale', None), ('upper lip', lm['stomion'][1] + 0.006),
             ('stomion', lm['stomion'][1]), ('lower lip', lm['stomion'][1] - 0.007), ('lip-chin fold', lm['stomion'][1] - 0.018),
             ('chin (pogonion)', lm['pogonion'][1]), ('chin bottom', lm['menton'][1] + 0.006)]
    for name, yv in feats:
        if name == 'nasion':
            y = m_nas[1]
        elif name == 'subnasale':
            y = m_sub[1]
        elif name == 'tip':
            y = m_tip[1]
        elif name == 'dorsum mid':
            y = 0.5 * (m_nas[1] + m_tip[1])
        else:
            p3 = front_at(yv)
            if p3 is None:
                continue
            y = prj(p3)[1]
        mm_ = M[np.argmin(np.abs(M[:, 1] - y))]
        if np.abs(C2[:, 1] - y).min() > 3:
            report.append(f'{name:16s}  (no client outline)'); continue
        cc = C2[np.argmin(np.abs(C2[:, 1] - y))]
        diff = (cc[0] - mm_[0]) * mm_px
        report.append(f'{name:16s} {diff:+5.1f} mm  (client {"ahead of" if diff > 0 else "behind"} the model)')
        d.line([(mm_[0], y), (cc[0], y)], fill=(255, 80, 80), width=1)
        d.text((min(mm_[0], cc[0]) - 60, y - 6), f'{diff:+.1f}', fill=(255, 255, 255))
    d.text((10, 10), 'blue = model outline   orange = client outline (aligned on nasion + subnasale)', fill=(255, 255, 255))
    crop = out.crop((int(m_nas[0] - 420), int(m_nas[1] - 330), int(m_nas[0] + 180), int(m_nas[1] + 520)))
    crop.save(os.path.join(it, 'profile_compare.jpg'), quality=92)
    open(os.path.join(it, 'profile_compare.txt'), 'w').write('\n'.join(report) + f'\n(mm per px {mm_px:.3f}, client->model scale {s:.3f}, rotation {np.degrees(ang):+.1f} deg)\n')
    print('\n'.join(report))


if __name__ == '__main__':
    main(sys.argv[1])
