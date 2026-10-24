# Face pose curation + local gap-fill (macOS)

**No uploads of your photos.** Optional network use is only public model weights (MediaPipe).

### Modes

1. **Real-only** — copy/select existing angles (`run_macos.sh` → `saved/`)
2. **Mild gap-fill** — small warps into empty bins (`run_macos_synth.sh` → `saved_diverse/`)
3. ~~Visible warp synth~~ — **deprecated** (distorts faces; do not use)
4. **Real-wide** — real-only scan of all `wcc1bex` photos (`run_macos_real_wide.sh` → `saved_real_wide/`)
5. **Video → side frames** — extract real side poses from an MP4 (`run_macos_video_sides.sh` → `saved_video_sides/`)
6. **Heavy local front→side** — LivePortrait on Apple Silicon (`heavy_3d/run_macos_liveportrait_sides.sh` → `saved_3d_sides/`) — see `heavy_3d/SETUP_AND_RUN.md`

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

### A) Real-wide (recommended after bad warp results)

Scans **all** of `wcc1bex` for real faces and keeps the best photo per angle bin. **No warping.**

```bash
cd ~/adpython/face_pose_curate
git pull origin cursor/face-pose-curate-071f
chmod +x run_macos_real_wide.sh
./run_macos_real_wide.sh
```

Cleanup bad warp folder (optional):

```bash
rm -rf "/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_visible"
```

**Why warps failed:** big 2D landmark warps cannot invent a true side view from a near-frontal aligned crop — they smear the face. Usable “new angles” need either more real photos or a real 3D-aware model (heavy local install), not amplify-warp.

### C) Front → side with LivePortrait (option B)

This **does** generate new side-ish views from your frontal photos (local, Apple Silicon):

```bash
cd ~/adpython/face_pose_curate
chmod +x heavy_3d/run_macos_liveportrait_sides.sh
./heavy_3d/run_macos_liveportrait_sides.sh
```

Details / limits: [`heavy_3d/SETUP_AND_RUN.md`](heavy_3d/SETUP_AND_RUN.md).

### A2) Mild gap-fill (subtle; often not worth it on `aligned`)

```bash
./run_macos_synth.sh
```

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
