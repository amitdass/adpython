# Real-only face pose curation (macOS, local)

Selects the widest **real** yaw/pitch coverage from your aligned faces.  
**No synthesis. No uploads of your photos.** The only optional network use is a one-time download of the public MediaPipe face-landmarker model.

## Your paths

| Role | Path |
|------|------|
| Input | `/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned` |
| Output | `/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved` |

The script **skips** the `saved` output folder when scanning input, so re-runs do not re-ingest previous results.

## Grid (locked)

- Yaw: **−75° … +75°**, step **5°**
- Pitch: **−25° … +25°**, step **5°**
- Roll: images with **\|roll\| > 15°** are skipped
- Keep **1** best real image per bin (override with `--per-bin`)

## Setup & run (on your Mac)

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
