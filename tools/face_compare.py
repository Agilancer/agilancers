"""Side-by-side likeness sheet: the client's video frames vs. the model at
matching angles. usage: face_compare.py <out.png>"""
import os, sys, subprocess, json
from PIL import Image, ImageDraw
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCR = sys.argv[2] if len(sys.argv) > 2 else '/tmp'
FRAMES = os.environ.get('HEADVID', '')
# (video frame, camera yaw in degrees: + = camera toward the character's left side)
PAIRS = [('f_001.jpg', 68), ('f_007.jpg', 38), ('f_015.jpg', 0), ('f_023.jpg', -32), ('f_031.jpg', -78)]


def main(out):
    views = []
    for f, yaw in PAIRS:
        views.append(f'cmp{yaw}')
    views.append('cmpclose')
    url = 'dev/preview.html?glb=../assets/models/agilancer_hero.glb&fov=48&hide=Arms&views=' + ','.join(views)
    tmp = os.path.join(SCR, 'cmp.png')
    subprocess.run(['node', 'tools/shot.cjs', url, tmp, '520', '600', '--noshot=1'], cwd=ROOT, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=600)
    cols = []
    for (f, yaw), v in zip(PAIRS, views):
        ref = Image.open(os.path.join(FRAMES, f)).convert('RGB').crop((180, 1400, 1980, 3476)).resize((520, 600))
        ren = Image.open(tmp.replace('.png', f'_{v}.png')).convert('RGB')
        col = Image.new('RGB', (520, 1200)); col.paste(ref, (0, 0)); col.paste(ren, (0, 600))
        cols.append(col)
    sheet = Image.new('RGB', (520 * len(cols), 1240), (20, 20, 24))
    for i, c in enumerate(cols):
        sheet.paste(c, (i * 520, 40))
    d = ImageDraw.Draw(sheet)
    d.text((10, 10), 'TOP ROW: the client (video frames)    BOTTOM ROW: the model at matching angles', fill=(255, 255, 255))
    sheet.resize((sheet.width * 2 // 3, sheet.height * 2 // 3)).save(out, quality=92)
    # large frontal pair for detail
    ref = Image.open(os.path.join(FRAMES, 'f_015.jpg')).convert('RGB').crop((330, 1700, 1290, 2980)).resize((600, 800))
    ren = Image.open(tmp.replace('.png', '_cmpclose.png')).convert('RGB').resize((600, 800))
    pair = Image.new('RGB', (1200, 800)); pair.paste(ref, (0, 0)); pair.paste(ren, (600, 0))
    pair.save(out.replace('.jpg', '_front.jpg'), quality=92)


if __name__ == '__main__':
    main(sys.argv[1])
