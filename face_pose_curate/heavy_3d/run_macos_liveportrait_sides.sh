#!/usr/bin/env bash
# Local LivePortrait: animate YOUR frontal face into side poses, extract stills.
# Photos stay on this Mac. Model weights download once (not your faces).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LP_DIR="${LP_DIR:-$HOME/LivePortrait}"
# Prefer curated real bins if present; otherwise the full aligned folder.
_DEFAULT_SRC="/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned"
if [[ -d "${_DEFAULT_SRC}/saved_real_wide" ]]; then
  _DEFAULT_SRC="${_DEFAULT_SRC}/saved_real_wide"
elif [[ -d "${_DEFAULT_SRC}/saved" ]] && find "${_DEFAULT_SRC}/saved" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | head -1 | grep -q .; then
  _DEFAULT_SRC="${_DEFAULT_SRC}/saved"
fi
SOURCE_DIR="${SOURCE_DIR:-$_DEFAULT_SRC}"
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
if ! command -v ffmpeg >/dev/null 2>&1 || ! command -v ffprobe >/dev/null 2>&1; then
  echo "ERROR: FFmpeg not found (LivePortrait needs ffmpeg + ffprobe)."
  echo "Install once, then re-run:"
  echo "  brew install ffmpeg"
  exit 1
fi
echo "FFmpeg OK: $(ffmpeg -version 2>/dev/null | head -1)"

# --- 1) clone LivePortrait ---
if [[ ! -d "$LP_DIR/.git" ]]; then
  echo "Cloning LivePortrait into $LP_DIR ..."
  git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git "$LP_DIR"
fi

cd "$LP_DIR"

# --- 2) venv + deps (Python 3.10/3.11 only — 3.12+ often breaks numpy==1.26.4 wheels) ---
pick_python() {
  for cand in python3.11 python3.10 python3.12; do
    if command -v "$cand" >/dev/null 2>&1; then
      ver="$("$cand" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
      major="${ver%%.*}"
      minor="${ver#*.}"
      if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 12 ]]; then
        echo "$cand"
        return 0
      fi
    fi
  done
  # fall back to python3 only if 3.10–3.12
  if command -v python3 >/dev/null 2>&1; then
    ver="$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')"
    major="${ver%%.*}"
    minor="${ver#*.}"
    if [[ "$major" -eq 3 && "$minor" -ge 10 && "$minor" -le 12 ]]; then
      echo "python3"
      return 0
    fi
  fi
  return 1
}

PY_BIN="$(pick_python || true)"
if [[ -z "${PY_BIN}" ]]; then
  echo "ERROR: LivePortrait needs Python 3.10, 3.11, or 3.12 (binary numpy wheels)."
  echo "Your default python3 is likely 3.13+, which tries to compile numpy and fails."
  echo
  echo "Install once, then re-run this script:"
  echo "  brew install python@3.11"
  echo "  # then either recreate venv, or:"
  echo "  rm -rf \"$LP_DIR/.venv\""
  echo "  export LP_PYTHON=python3.11"
  exit 1
fi
if [[ -n "${LP_PYTHON:-}" ]]; then
  PY_BIN="$LP_PYTHON"
fi

echo "Using interpreter: $PY_BIN ($("$PY_BIN" -V 2>&1))"

# Recreate venv if it was made with a bad Python (e.g. 3.13)
if [[ -d .venv ]]; then
  VENV_VER="$(.venv/bin/python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")' 2>/dev/null || echo bad)"
  V_MAJOR="${VENV_VER%%.*}"
  V_MINOR="${VENV_VER#*.}"
  if [[ "$VENV_VER" == "bad" || "$V_MAJOR" -ne 3 || "$V_MINOR" -lt 10 || "$V_MINOR" -gt 12 ]]; then
    echo "Removing incompatible venv (was Python $VENV_VER)..."
    rm -rf .venv
  fi
fi

if [[ ! -d .venv ]]; then
  echo "Creating LivePortrait venv with $PY_BIN ..."
  "$PY_BIN" -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

echo "Installing numpy wheel first (never compile from source)..."
pip install --only-binary=:all: "numpy==1.26.4" || pip install --only-binary=:all: "numpy>=1.26.4,<2"
# InsightFace vendored code imports requests; not always in requirements_macOS.txt
pip install "requests>=2.31,<3" "huggingface_hub>=0.20"

if [[ -f requirements_macOS.txt ]]; then
  # Prefer wheels; skip rebuilds of numpy
  pip install --only-binary=numpy -r requirements_macOS.txt
else
  echo "requirements_macOS.txt missing — installing requirements.txt (may need tweaks)"
  pip install --only-binary=numpy -r requirements.txt || true
  pip install onnxruntime-silicon || pip install onnxruntime || true
fi
# Ensure again after requirements (in case a resolver removed it)
pip install "requests>=2.31,<3"

# --- 3) pretrained weights (required) ---
WEIGHT_FILE="pretrained_weights/liveportrait/base_models/appearance_feature_extractor.pth"
download_weights() {
  echo "Downloading LivePortrait pretrained weights (public models only; not your photos)..."
  pip install -U "huggingface_hub"
  # Avoid HF Xet transport bugs ("Byte range not sequential")
  export HF_HUB_DISABLE_XET=1
  export HF_HUB_ENABLE_HF_TRANSFER=0
  # Newer hub versions expose `hf` (not always `huggingface-cli`)
  if command -v hf >/dev/null 2>&1; then
    HF_HUB_DISABLE_XET=1 hf download KlingTeam/LivePortrait \
      --local-dir pretrained_weights \
      --exclude "*.git*" "README.md" "docs" || return 1
  elif command -v huggingface-cli >/dev/null 2>&1; then
    HF_HUB_DISABLE_XET=1 huggingface-cli download KlingTeam/LivePortrait \
      --local-dir pretrained_weights \
      --exclude "*.git*" "README.md" "docs" || return 1
  else
    HF_HUB_DISABLE_XET=1 python - <<'PY' || return 1
import os
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="KlingTeam/LivePortrait",
    local_dir="pretrained_weights",
    ignore_patterns=["*.git*", "README.md", "docs", "docs/*"],
)
print("snapshot_download finished")
PY
  fi
  return 0
}

if [[ ! -f "$WEIGHT_FILE" ]]; then
  download_weights || true
fi
if [[ ! -f "$WEIGHT_FILE" ]]; then
  echo "Primary HF download incomplete; trying hf-mirror..."
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  download_weights || true
fi
if [[ ! -f "$WEIGHT_FILE" ]]; then
  echo "ERROR: Missing $WEIGHT_FILE"
  echo "Download weights manually, then re-run:"
  echo "  cd ~/LivePortrait"
  echo "  source .venv/bin/activate"
  echo "  pip install -U \"huggingface_hub[cli]\""
  echo "  huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude \"*.git*\" \"README.md\" \"docs\""
  echo "Or if HF is blocked:"
  echo "  export HF_ENDPOINT=https://hf-mirror.com"
  echo "  huggingface-cli download KlingTeam/LivePortrait --local-dir pretrained_weights --exclude \"*.git*\" \"README.md\" \"docs\""
  exit 1
fi
echo "Weights OK: $WEIGHT_FILE"

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
_SRC_LIST="$(mktemp)"
find "$SOURCE_DIR" -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) 2>/dev/null | head -n "$SOURCE_MAX" > "$_SRC_LIST" || true
while IFS= read -r line; do
  [[ -n "$line" ]] && SOURCES+=("$line")
done < "$_SRC_LIST"
rm -f "$_SRC_LIST"
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
