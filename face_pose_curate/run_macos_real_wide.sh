#!/usr/bin/env bash
# Real-only: scan ALL images under wcc1bex, keep best real face per yaw/pitch bin.
# No synthesis (no warps). Photos never leave the Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
INPUT="${INPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex}"
OUTPUT="${OUTPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_real_wide}"

cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "============================================================"
echo "REAL-ONLY wide scan (recommended after bad warp synth)"
echo "  Input:  $INPUT"
echo "  Output: $OUTPUT"
echo "  No generated/warped faces — only real photos, sorted by angle."
echo "============================================================"
echo

# Exclude prior tool outputs inside the tree via output path + name filters in Python
python curate_real_only.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --per-bin 1 \
  --dry-run

echo
read -r -p "Dry run done. Copy best REAL images into output bins? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  python curate_real_only.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --per-bin 1
  echo "Done. Open: $OUTPUT"
  echo "You can delete bad warp outputs:"
  echo "  rm -rf \".../aligned/saved_visible\""
else
  echo "Stopped after dry run."
fi
