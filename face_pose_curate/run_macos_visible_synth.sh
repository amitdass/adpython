#!/usr/bin/env bash
# Stronger, visibly different face-angle synthesis from a wide local source tree.
# Photos never leave the Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
INPUT="${INPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex}"
OUTPUT="${OUTPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_visible}"

cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Input root (recursive): $INPUT"
echo "Output:                 $OUTPUT"
echo "This creates NEW turned faces (amplify=2, yaw offsets ±20..±45)."
echo "Prior mild synths in saved_diverse/ are left untouched."
echo

python synthesize_visible_angles.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --max-sources 120 \
  --amplify 2.0 \
  --min-yaw-delta 12 \
  --yaw-offsets "-45,-30,-20,20,30,45" \
  --pitch-offsets "-12,0,12" \
  --dry-run

echo
read -r -p "Dry run OK? Write new angle images now? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  python synthesize_visible_angles.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --max-sources 120 \
    --amplify 2.0 \
    --min-yaw-delta 12 \
    --yaw-offsets "-45,-30,-20,20,30,45" \
    --pitch-offsets "-12,0,12"
  echo
  echo "Open new images:"
  echo "  open \"$OUTPUT/synth_angles\""
else
  echo "Stopped after dry run."
fi
