#!/usr/bin/env bash
# Local LivePortrait: animate YOUR frontal face into side poses, extract stills.
# Photos stay on this Mac. Model weights download once (not your faces).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LP_DIR="${LP_DIR:-$HOME/LivePortrait}"
SOURCE_DIR="${SOURCE_DIR:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved}"
OUTPUT="${OUTPUT:-/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_3d_sides}"
SOURCE_MAX="${SOURCE_MAX:-5}"
EVERY_N="${EVERY_N:-2}"
TOP_N="${TOP_N:-24}"

echo "============================================================"
echo "Heavy local side-pose generation (LivePortrait on Mac)"
echo "  LivePortrait: $LP_DIR"
echo "  Sources:      $SOURCE_DIR  (up to $SOURCE_MAX images)"
echo "  Output:       $OUTPUT"
echo "============================================================"

# --- 0) basic checks ---
if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This script is for macOS."
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "WARNING: Apple Silicon (arm64) is strongly recommended. Intel is untested."
fi

# --- 1) clone LivePortrait ---
if [[ ! -d "$LP_DIR/.git" ]]; then
  echo "Cloning LivePortrait into $LP_DIR ..."
  git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git "$LP_DIR"
fi

cd "$LP_DIR"

# --- 2) venv + deps ---
if [[ ! -d .venv ]]; then
  echo "Creating LivePortrait venv (python3.10/3.11 preferred)..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

if [[ -f requirements_macOS.txt ]]; then
  pip install -r requirements_macOS.txt
else
  echo "requirements_macOS.txt missing — installing requirements.txt (may need tweaks)"
  pip install -r requirements.txt || true
  pip install onnxruntime-silicon || pip install onnxruntime || true
fi

# --- 3) pretrained weights ---
if [[ ! -d pretrained_weights/insightface ]] && [[ ! -d pretrained_weights/liveportrait ]]; then
  echo "Downloading pretrained weights (one-time, public models)..."
  if command -v huggingface-cli >/dev/null 2>&1; then
    huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "docs/*" || true
  fi
  # Official helper if present
  if [[ -f scripts/download_models.py ]]; then
    python scripts/download_models.py || true
  elif [[ -f inference.py ]] && [[ ! -d pretrained_weights ]]; then
    echo "Trying git-lfs / README download path..."
    mkdir -p pretrained_weights
    echo "If inference fails on missing weights, follow LivePortrait README 'Download pretrained weights'."
  fi
fi

DRIVING="${DRIVING:-$LP_DIR/assets/examples/driving/d0.mp4}"
if [[ ! -f "$DRIVING" ]]; then
  # fallback: first mp4 under assets
  DRIVING="$(find "$LP_DIR/assets" -name '*.mp4' 2>/dev/null | head -1 || true)"
fi
if [[ -z "${DRIVING}" || ! -f "$DRIVING" ]]; then
  echo "No driving video found under LivePortrait assets."
  echo "Set DRIVING=/path/to/head_turn.mp4 and re-run."
  exit 1
fi
echo "Driving pose video: $DRIVING"

# --- 4) pick source images (prefer fewer, sharper files) ---
mkdir -p "$OUTPUT/sources_used" "$OUTPUT/animations" "$OUTPUT/side"
SOURCES=()
while IFS= read -r line; do
  SOURCES+=("$line")
done < <(find "$SOURCE_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | head -n "$SOURCE_MAX")
if [[ ${#SOURCES[@]} -eq 0 ]]; then
  echo "No images found in SOURCE_DIR=$SOURCE_DIR"
  echo "Set SOURCE_DIR to a folder of your face photos (near-frontal works best)."
  exit 1
fi

echo "Using ${#SOURCES[@]} source image(s):"
for s in "${SOURCES[@]}"; do
  echo "  $s"
done

# --- 5) run LivePortrait per source ---
export PYTORCH_ENABLE_MPS_FALLBACK=1
for SRC in "${SOURCES[@]}"; do
  base="$(basename "$SRC")"
  stem="${base%.*}"
  cp "$SRC" "$OUTPUT/sources_used/$base" || true
  echo
  echo ">>> Animating $base ..."
  # Output lands in LivePortrait animations/ by default; we copy afterward
  python inference.py -s "$SRC" -d "$DRIVING" || {
    echo "inference.py failed for $base — check weights/RAM. Continuing if others work."
    continue
  }
  # Copy newest animation mp4
  newest="$(ls -t animations/*.mp4 2>/dev/null | head -1 || true)"
  if [[ -n "$newest" ]]; then
    cp "$newest" "$OUTPUT/animations/${stem}__driven.mp4"
    echo "Saved animation: $OUTPUT/animations/${stem}__driven.mp4"
  fi
done

# --- 6) extract most side-like stills from animations ---
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate" 2>/dev/null || true
# Prefer tool venv for our extractor (mediapipe); fall back to LP venv if needed
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
  pip install -q -r "$ROOT/requirements.txt" || true
fi

shopt -s nullglob
ANIMS=("$OUTPUT/animations"/*.mp4)
if [[ ${#ANIMS[@]} -eq 0 ]]; then
  echo "No animation mp4s produced. Fix LivePortrait inference first (weights/RAM)."
  exit 1
fi

for ANIM in "${ANIMS[@]}"; do
  stem="$(basename "$ANIM" .mp4)"
  out_sub="$OUTPUT/side/$stem"
  echo "Extracting side stills from $ANIM ..."
  python "$ROOT/extract_side_poses_from_video.py" \
    --video "$ANIM" \
    --output "$out_sub" \
    --every-n "$EVERY_N" \
    --top-n "$TOP_N" \
    --min-yaw 8 || true
done

echo
echo "DONE."
echo "  Animations: $OUTPUT/animations"
echo "  Side stills: $OUTPUT/side/<clip>/side/"
echo "  Debug labels: $OUTPUT/side/<clip>/debug_labeled/"
echo
echo "Open:"
echo "  open \"$OUTPUT/side\""
echo
echo "Tip: for finer yaw control, run Gradio:"
echo "  cd \"$LP_DIR\" && source .venv/bin/activate"
echo "  PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py"
