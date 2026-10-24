# Heavy local 3D-ish side poses (Mac) — LivePortrait

This **does** turn a frontal (or near-frontal) photo of you into **new side / three-quarter views**.  
Everything runs on your Mac. Your face images are not uploaded. Only public model weights are downloaded once.

## What you need

- **Apple Silicon** Mac (M1/M2/M3/M4). Intel is not really supported.
- **16 GB RAM minimum**, 24–32 GB much happier
- Disk: ~5–10 GB for code + models
- Time: first setup can take a while; each image is slower than on NVIDIA

## What it does

1. Takes **your** face photo as source  
2. Uses a **driving** clip/image that has a head turn (pose only — not their identity)  
3. Animates your face into those poses  
4. We extract still frames that look most side-on  

This is the realistic “front → side” path. Old 2D warps are not used.

## One-command setup + run

```bash
cd ~/adpython/face_pose_curate
git pull origin cursor/face-pose-curate-071f
chmod +x heavy_3d/run_macos_liveportrait_sides.sh
./heavy_3d/run_macos_liveportrait_sides.sh
```

Defaults:

| | Path |
|--|------|
| Source faces | `.../aligned/saved` (your curated reals) — or set `SOURCE_DIR` |
| LivePortrait install | `~/LivePortrait` |
| Output sides | `.../aligned/saved_3d_sides` |

Optional env overrides:

```bash
export SOURCE_DIR="/path/to/folder/of/your/face/jpgs"
export SOURCE_MAX=5          # how many source faces to animate (quality over quantity)
export DRIVING="$HOME/LivePortrait/assets/examples/driving/d0.mp4"
export OUTPUT="/Users/amdass/Documents/Cisco Icons/wcc1bex/untled folder/WebEx/data_src/aligned/saved_3d_sides"
./heavy_3d/run_macos_liveportrait_sides.sh
```

## Manual Gradio (precise yaw)

After setup:

```bash
cd ~/LivePortrait
source .venv/bin/activate
PYTORCH_ENABLE_MPS_FALLBACK=1 python app.py
```

In the UI: load your photo → use **pose editing** / driving to push yaw toward profile → export.  
Fewer images, often cleaner control.

## Honest limits

- Hair, glasses, ears, extreme profile can look off  
- Not forensic-perfect identity  
- Much better than 2D warp for “looks like a real turn”  
- If RAM pressure hits, close other apps or lower `SOURCE_MAX` to 1–2  

## Privacy

- Inference is local  
- Do **not** pass `--share` to Gradio  
- Model download is weights only (Hugging Face), not your photos  
