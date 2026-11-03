#!/usr/bin/env bash
# One likeness iteration: refit the MakeHuman face to the targets in
# tools/build/hero/face_tweaks.json, rebuild (re-baking only head + hair) and
# render the comparison sheets.   usage: tools/face_iter.sh <outdir>
set -e
cd "$(dirname "$0")/.."
OUT=$1; mkdir -p "$OUT"
PY=${PY:-python3}
$PY -I tools/build/hero/mh_landmarks.py > /dev/null 2>&1 || true
$PY -I tools/build/hero/fit_face.py | grep -E "rms|active" > "$OUT/fit.txt" || true
$PY -I tools/build/hero/geometry_cache.py head bun > /dev/null 2>&1
$PY -I tools/build/hero/mh_landmarks.py > /dev/null
REBAKE=Head,Hair $PY -I tools/build/hero/build_hero.py > "$OUT/build.log" 2>&1
HEADVID=${HEADVID:?} python3 tools/face_compare.py "$OUT/compare.jpg" "$OUT"
HEADVID=$HEADVID python3 tools/nose_compare.py "$OUT"
HEADVID=$HEADVID python3 tools/nose_compare.py "$OUT" chin
HEADVID=$HEADVID python3 tools/profile_compare.py "$OUT" > /dev/null
cp tools/build/hero/face_tweaks.json "$OUT/"
echo "iteration rendered: $OUT"
