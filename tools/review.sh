#!/usr/bin/env bash
# Render the review package used by the critics: tools/review.sh <outdir>
set -e
OUT=${1:-build/review}; mkdir -p "$OUT"
ONLY=${ONLY:-turn close heavy dash jump wire arms}   # e.g. ONLY="heavy dash" to re-render some sections
want() { [[ " $ONLY " == *" $1 "* ]]; }
cd "$(dirname "$0")/.."
shot() { timeout 900 node tools/shot.cjs "$1" "$OUT/$2.png" ${3:-560} ${4:-700} --noshot=1 --timeout=800000 >/dev/null 2>&1 || echo "shot failed: $2"; }
want turn && shot "index.html?gl=1&capture&anim=Idle_Shield_Loop&settle=2.5&views=front,q34,left,back,q34b" turn
want close && shot "index.html?gl=1&capture&anim=Idle_Shield_Loop&settle=2.5&views=face,face34,headback,torso" close 600 600
V=""; for i in 1 2 3 4 5 6 7 8; do v=q34b; [ $i -gt 4 ] && v=back; [ $i -gt 6 ] && v=q34; V="$V,adv0.38_$v~$i"; done
want heavy && shot "index.html?gl=1&capture&anim=Sword_Heavy_Combo&settle=0.8&views=${V#,}" heavy 480 600
V=""; for i in 1 2 3 4 5 6; do V="$V,adv0.22_q34l~$i"; done
want dash && shot "index.html?gl=1&capture&rm&follow&anim=Sword_Dash&settle=0.3&views=${V#,}" dash 480 600
V=""; for i in 1 2 3 4 5 6; do V="$V,adv0.3_right~$i"; done
want jump && shot "index.html?gl=1&capture&rm&follow&anim=NinjaJump_Start&settle=0.2&views=${V#,}" jump 480 600
want wire && shot "index.html?gl=1&capture&anim=A_TPose&tpose&settle=0.5&wire&views=front,q34" wire 700 700
want arms && shot "dev/armsheet.html" arms 1400 700
python3 - "$OUT" <<'PY'
import sys, glob, os, re
from PIL import Image
out = sys.argv[1]
def sheet(prefix, name, cols, scale=1.0):
    fs = glob.glob(os.path.join(out, prefix + '_*.png'))
    def key(f):
        m = re.search(r'~(\d+)', f); return (int(m.group(1)) if m else 0, f)
    fs.sort(key=key)
    if not fs: return
    ims = [Image.open(f).convert('RGB') for f in fs]
    w, h = ims[0].size; rows = (len(ims) + cols - 1) // cols
    o = Image.new('RGB', (w * cols, h * rows), (30, 30, 34))
    for i, im in enumerate(ims): o.paste(im, ((i % cols) * w, (i // cols) * h))
    if scale != 1: o = o.resize((int(o.width * scale), int(o.height * scale)))
    o.save(os.path.join(out, name + '.png'))
    for f in fs: os.remove(f)
sheet('turn', '01_turnaround', 5, 0.7)
sheet('close', '02_closeups', 4, 0.75)
sheet('heavy', '03_cloth_heavy_combo', 4, 0.8)
sheet('dash', '04_cloth_sword_dash_rootmotion', 6, 0.7)
sheet('jump', '05_cloth_ninja_jump_rootmotion', 6, 0.7)
sheet('wire', '06_wireframe_tpose', 2, 0.8)
sheet('arms', '07_lances_shields', 1, 0.9)
PY
ls -la "$OUT"
