# Face pose curation + local gap-fill (macOS)

**No uploads of your photos.** Optional network use is only public model weights (MediaPipe).

### Modes

1. **Real-only** — copy/select existing angles (`run_macos.sh` → `saved/`)
2. **Real + synth** — also **generate new views** by warping aligned faces into missing bins (`run_macos_synth.sh` → `saved_diverse/`)

## Your paths

| Role | Path |
|------|------|
| Input | `/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned` |
| Real-only output | `.../aligned/saved` |
| Diverse output | `.../aligned/saved_diverse` (`real/` + `synth/`) |

The script **skips** the `saved` output folder when scanning input, so re-runs do not re-ingest previous results.

## Grid (locked)

- Yaw: **−75° … +75°**, step **5°**
- Pitch: **−25° … +25°**, step **5°**
- Roll: images with **\|roll\| > 45°** are skipped (aligned faces are usually near 0°)
- Keep **1** best real image per bin (override with `--per-bin`)

### Re-run after a bad first pass

If an earlier run only filled a handful of bins (e.g. almost everything skipped as `roll`), pull the latest script, remove the old output, and run again:

```bash
rm -rf "/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved"
git pull
./run_macos.sh
```

**Note:** DeepFaceLab `aligned` crops are warped toward frontal, so extreme yaw/pitch coverage will be lower than in raw frames. For wider true angles, point `--input` at unaligned source frames later.

## Setup & run (on your Mac)

### A) Generate new angles (recommended next step)

Uses **all** images in `aligned` as sources. For each empty yaw/pitch bin, warps the nearest real face toward that pose (max **20° yaw** / **12° pitch** from a real source so quality stays usable).

```bash
cd ~/adpython/face_pose_curate   # or your clone path
git pull origin cursor/face-pose-curate-071f
chmod +x run_macos_synth.sh
./run_macos_synth.sh
```

Output:

- `saved_diverse/real/` — selected real faces  
- `saved_diverse/synth/` — **new** warped images  
- `manifest.csv` — `kind=real|synth` plus source pose  
- `coverage_heatmap.png` — 0=empty, 1=real, 2=synth  

Synths are **not** copies: they are new pixels. They still use your face texture, so identity stays close; large angle jumps are blocked on purpose.

### B) Real-only (previous step)

```bash
cd /path/to/face_pose_curate
chmod +x run_macos.sh
./run_macos.sh
```

Or manually:

```bash
cd /path/to/face_pose_curate
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Dry run (coverage only)
python curate_real_only.py \
  --input "/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned" \
  --output "/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved" \
  --dry-run

# Full run (copy best images into bins)
python curate_real_only.py \
  --input "/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned" \
  --output "/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved"
```

If the model download is blocked, fetch this file in a browser and pass `--model`:

`https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`

## Outputs

- `saved/yaw_+020.0_pitch_-005.0/…` — real images per bin  
- `saved/manifest.csv` — source path, measured pose, bin, scores  
- `saved/coverage.csv` — every grid cell: filled or empty  
- `saved/coverage_heatmap.png` — visual map of coverage  

Empty bins mean that angle was **not present** in your source set (expected for real-only).

## Notes

- Confirm the folder name spelling: `untled folder` vs `untitled folder`.
- Prefer keeping originals elsewhere; `saved` is a curated subset (copies by default).
- Use `--copy-mode symlink` to avoid duplicating disk use.
