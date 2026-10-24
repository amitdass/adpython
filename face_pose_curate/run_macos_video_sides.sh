#!/usr/bin/env bash
# Extract sharp SIDE / three-quarter face frames from a local video. No uploads.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VIDEO="${VIDEO:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/14bler.com.mp4}"
OUTPUT="${OUTPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_video_sides}"

cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating venv..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

if [[ ! -f "$VIDEO" ]]; then
  echo "Video not found: $VIDEO"
  exit 1
fi

echo "Video:  $VIDEO"
echo "Output: $OUTPUT"
echo "Keeping frames with |yaw| >= 22° (side / three-quarter), best per 5° bin."
echo

python extract_side_poses_from_video.py \
  --video "$VIDEO" \
  --output "$OUTPUT" \
  --every-n 3 \
  --min-yaw 22 \
  --also-frontal \
  --dry-run

echo
read -r -p "Dry run OK? Extract side frames now? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  python extract_side_poses_from_video.py \
    --video "$VIDEO" \
    --output "$OUTPUT" \
    --every-n 3 \
    --min-yaw 22 \
    --also-frontal
  echo
  echo "Open sides:"
  echo "  open \"$OUTPUT/side\""
else
  echo "Stopped after dry run."
fi
