#!/usr/bin/env bash
# Local pose gap-fill: real copies + newly warped views. Photos never leave the Mac.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
INPUT="${INPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned}"
OUTPUT="${OUTPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_diverse}"

cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Input:  $INPUT  (all aligned faces used as sources)"
echo "Output: $OUTPUT  (real/ + synth/)"
echo "Synths are NEW warped images (not copies). Identity stays tied to source texture."
echo

python synthesize_pose_gaps.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --max-delta-yaw 20 \
  --max-delta-pitch 12 \
  --dry-run

echo
read -r -p "Dry run done. Write real copies + synth images? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  python synthesize_pose_gaps.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --max-delta-yaw 20 \
    --max-delta-pitch 12
  echo "Done."
  echo "  $OUTPUT/real/   = selected real faces"
  echo "  $OUTPUT/synth/  = newly generated views"
  echo "  $OUTPUT/manifest.csv"
  echo "  $OUTPUT/coverage_heatmap.png"
else
  echo "Stopped after dry run."
fi
