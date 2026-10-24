#!/usr/bin/env bash
# Real-only pose curation — runs entirely on your Mac. Photos never leave the machine.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
INPUT="${INPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned}"
OUTPUT="${OUTPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved}"

cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Input:  $INPUT"
echo "Output: $OUTPUT"
echo

# 1) Dry run first — reports only, no copies
python curate_real_only.py \
  --input "$INPUT" \
  --output "$OUTPUT" \
  --per-bin 1 \
  --dry-run

echo
read -r -p "Dry run done. Copy selected images into output bins? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  python curate_real_only.py \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --per-bin 1
  echo "Done. See $OUTPUT/manifest.csv and coverage_heatmap.png"
else
  echo "Stopped after dry run. Re-run without answering N, or run curate_real_only.py without --dry-run."
fi
