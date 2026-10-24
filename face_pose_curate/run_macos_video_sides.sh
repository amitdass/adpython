#!/usr/bin/env bash
# Extract the most side-like face frames from a local video. No uploads.
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
echo "Keeps top-40 highest |yaw| frames + labeled debug timeline."
echo

python extract_side_poses_from_video.py \
  --video "$VIDEO" \
  --output "$OUTPUT" \
  --every-n 2 \
  --top-n 40 \
  --min-yaw 8 \
  --dry-run

echo
read -r -p "Dry run OK? Export frames + debug labels now? [y/N] " ans
if [[ "${ans:-}" =~ ^[Yy]$ ]]; then
  rm -rf "$OUTPUT"
  python extract_side_poses_from_video.py \
    --video "$VIDEO" \
    --output "$OUTPUT" \
    --every-n 2 \
    --top-n 40 \
    --min-yaw 8
  echo
  echo "1) Check labeled timeline (yaw printed on image):"
  echo "   open \"$OUTPUT/debug_labeled\""
  echo "2) Then check selected sides:"
  echo "   open \"$OUTPUT/side\""
else
  echo "Stopped after dry run."
fi
