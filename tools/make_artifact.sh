#!/usr/bin/env bash
# Prepare the viewer for hosts that do not serve .glb and limit file sizes:
# models -> base64 text split into <= 8 MB parts (main.js joins "<file>.b64.N.txt").
# usage: tools/make_artifact.sh [outdir=build/artifact]
set -e
cd "$(dirname "$0")/.."
OUT=${1:-build/artifact}
mkdir -p "$OUT/models"
rm -f "$OUT"/models/*.b64*.txt
for f in agilancer_hero.glb agilancer_arms.glb UAL2_Standard_RM.glb; do
  base64 -w 0 "assets/models/$f" | split -b 8000000 -d -a 1 - "$OUT/models/$f.b64."
  for p in "$OUT"/models/$f.b64.?; do mv "$p" "$p.txt"; done
done
ls -la "$OUT/models"
