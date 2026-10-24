#!/usr/bin/env python3
"""
Real-only face pose curation (local, no uploads).

Scans aligned face images, estimates yaw/pitch/roll, bins them on a fixed grid,
keeps the best real image(s) per bin, and writes a local dataset + coverage report.

Never sends images to the network. Optional one-time download is only the
MediaPipe face-landmarker model weights (not your photos).
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
# Locked defaults from the real-only plan
DEFAULT_YAW_MIN = -75.0
DEFAULT_YAW_MAX = 75.0
DEFAULT_PITCH_MIN = -25.0
DEFAULT_PITCH_MAX = 25.0
DEFAULT_STEP = 5.0
DEFAULT_MAX_ROLL = 15.0
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Canonical 3D face model points (mm-ish), paired with MediaPipe landmark indices
MODEL_POINTS_3D = np.array(
    [
        [0.0, 0.0, 0.0],  # nose tip
        [0.0, -330.0, -65.0],  # chin
        [-225.0, 170.0, -135.0],  # left eye outer
        [225.0, 170.0, -135.0],  # right eye outer
        [-150.0, -150.0, -125.0],  # left mouth
        [150.0, -150.0, -125.0],  # right mouth
    ],
    dtype=np.float64,
)
LANDMARK_IDS = (1, 152, 33, 263, 61, 291)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


@dataclass
class FaceSample:
    path: Path
    yaw: float
    pitch: float
    roll: float
    sharpness: float
    face_frac: float
    score: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Curate a real-only yaw/pitch face dataset (local only)."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder of aligned face images (not scanned recursively into --output).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output folder for binned copies + reports.",
    )
    p.add_argument("--yaw-min", type=float, default=DEFAULT_YAW_MIN)
    p.add_argument("--yaw-max", type=float, default=DEFAULT_YAW_MAX)
    p.add_argument("--pitch-min", type=float, default=DEFAULT_PITCH_MIN)
    p.add_argument("--pitch-max", type=float, default=DEFAULT_PITCH_MAX)
    p.add_argument("--step", type=float, default=DEFAULT_STEP)
    p.add_argument(
        "--max-roll",
        type=float,
        default=DEFAULT_MAX_ROLL,
        help="Skip images whose |roll| exceeds this (degrees).",
    )
    p.add_argument(
        "--per-bin",
        type=int,
        default=1,
        help="How many best real images to keep per yaw/pitch bin.",
    )
    p.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to face_landmarker.task (downloaded once if missing).",
    )
    p.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to place selected files into output bins.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate poses and print coverage; write no image copies.",
    )
    return p.parse_args()


def ensure_model(model_path: Path) -> Path:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if model_path.is_file() and model_path.stat().st_size > 0:
        return model_path
    print(f"Downloading MediaPipe model (one-time) to {model_path} ...")
    print("Your photos are NOT uploaded; only public model weights are fetched.")
    urllib.request.urlretrieve(MODEL_URL, model_path)
    return model_path


def iter_images(root: Path, exclude_dirs: set[Path]) -> Iterable[Path]:
    root = root.resolve()
    exclude_resolved = {d.resolve() for d in exclude_dirs}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            continue
        resolved = path.resolve()
        # Skip anything inside excluded dirs (e.g. previous output under input)
        skipped = False
        for ex in exclude_resolved:
            try:
                resolved.relative_to(ex)
                skipped = True
                break
            except ValueError:
                continue
        if not skipped:
            yield path


def bin_center(value: float, vmin: float, step: float) -> float:
    idx = round((value - vmin) / step)
    return vmin + idx * step


def in_range(value: float, vmin: float, vmax: float, step: float) -> bool:
    c = bin_center(value, vmin, step)
    return vmin - 1e-6 <= c <= vmax + 1e-6


def rotation_matrix_to_euler(r: np.ndarray) -> tuple[float, float, float]:
    """Return yaw, pitch, roll in degrees (yaw=left/right, pitch=up/down)."""
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.degrees(math.atan2(-r[2, 0], sy))
        yaw = math.degrees(math.atan2(r[1, 0], r[0, 0]))
        roll = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    else:
        pitch = math.degrees(math.atan2(-r[2, 0], sy))
        yaw = math.degrees(math.atan2(-r[0, 1], r[1, 1]))
        roll = 0.0
    # MediaPipe/OpenCV camera convention → intuitive face pose:
    # positive yaw = subject turned toward their left (camera sees right cheek)
    return yaw, pitch, roll


def estimate_pose_from_landmarks(
    landmarks, width: int, height: int
) -> tuple[float, float, float] | None:
    image_points = []
    for idx in LANDMARK_IDS:
        lm = landmarks[idx]
        image_points.append([lm.x * width, lm.y * height])
    image_points = np.array(image_points, dtype=np.float64)

    focal = width
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array(
        [[focal, 0, center[0]], [0, focal, center[1]], [0, 0, 1]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    ok, rvec, _tvec = cv2.solvePnP(
        MODEL_POINTS_3D,
        image_points,
        camera_matrix,
        dist,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        return None
    rot, _ = cv2.Rodrigues(rvec)
    return rotation_matrix_to_euler(rot)


def sharpness_score(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def face_fraction_from_landmarks(landmarks, width: int, height: int) -> float:
    xs = [lm.x * width for lm in landmarks]
    ys = [lm.y * height for lm in landmarks]
    bw = max(xs) - min(xs)
    bh = max(ys) - min(ys)
    area = max(bw, 1.0) * max(bh, 1.0)
    return float(area / float(width * height))


def build_landmarker(model_path: Path) -> mp_vision.FaceLandmarker:
    base = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = mp_vision.FaceLandmarkerOptions(
        base_options=base,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return mp_vision.FaceLandmarker.create_from_options(options)


def analyze_image(
    path: Path, landmarker: mp_vision.FaceLandmarker
) -> FaceSample | None:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    h, w = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    pose = estimate_pose_from_landmarks(lms, w, h)
    if pose is None:
        return None
    yaw, pitch, roll = pose
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    sharp = sharpness_score(gray)
    frac = face_fraction_from_landmarks(lms, w, h)
    # Prefer sharp, large faces; mild penalty for heavy roll
    score = sharp * (0.5 + 0.5 * min(frac, 1.0)) / (1.0 + abs(roll) / 45.0)
    return FaceSample(path, yaw, pitch, roll, sharp, frac, score)


def bin_key(yaw: float, pitch: float, args: argparse.Namespace) -> tuple[float, float]:
    return (
        bin_center(yaw, args.yaw_min, args.step),
        bin_center(pitch, args.pitch_min, args.step),
    )


def folder_name(yaw: float, pitch: float) -> str:
    # Compact, sortable names: yaw_+020.0_pitch_-005.0
    return f"yaw_{yaw:+06.1f}_pitch_{pitch:+06.1f}"


def write_coverage_csv(
    path: Path,
    bins: dict[tuple[float, float], list[FaceSample]],
    args: argparse.Namespace,
) -> None:
    yaw_vals = []
    v = args.yaw_min
    while v <= args.yaw_max + 1e-9:
        yaw_vals.append(round(v, 6))
        v += args.step
    pitch_vals = []
    v = args.pitch_min
    while v <= args.pitch_max + 1e-9:
        pitch_vals.append(round(v, 6))
        v += args.step

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["yaw", "pitch", "count", "status"])
        for pitch in pitch_vals:
            for yaw in yaw_vals:
                n = len(bins.get((yaw, pitch), []))
                status = "filled" if n else "empty"
                w.writerow([yaw, pitch, n, status])


def write_heatmap_png(
    path: Path,
    bins: dict[tuple[float, float], list[FaceSample]],
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    yaw_vals = []
    v = args.yaw_min
    while v <= args.yaw_max + 1e-9:
        yaw_vals.append(round(v, 6))
        v += args.step
    pitch_vals = []
    v = args.pitch_min
    while v <= args.pitch_max + 1e-9:
        pitch_vals.append(round(v, 6))
        v += args.step

    grid = np.zeros((len(pitch_vals), len(yaw_vals)), dtype=np.float64)
    for yi, yaw in enumerate(yaw_vals):
        for pi, pitch in enumerate(pitch_vals):
            grid[pi, yi] = len(bins.get((yaw, pitch), []))

    fig, ax = plt.subplots(figsize=(12, 4.5))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[
            args.yaw_min - args.step / 2,
            args.yaw_max + args.step / 2,
            args.pitch_min - args.step / 2,
            args.pitch_max + args.step / 2,
        ],
        interpolation="nearest",
    )
    ax.set_xlabel("Yaw (deg)")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Real-only pose coverage (count per bin)")
    fig.colorbar(im, ax=ax, label="images kept")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def place_file(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    model_path = args.model or (Path(__file__).resolve().parent / "models" / "face_landmarker.task")
    try:
        ensure_model(model_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not fetch model automatically ({exc}).\n"
            f"Download manually (browser is fine) and pass --model PATH:\n  {MODEL_URL}",
            file=sys.stderr,
        )
        return 1

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    landmarker = build_landmarker(model_path)
    exclude = {output_dir}
    images = list(iter_images(input_dir, exclude))
    print(f"Found {len(images)} images under {input_dir}")

    accepted: list[FaceSample] = []
    skipped = defaultdict(int)

    for i, path in enumerate(images, 1):
        if i % 50 == 0 or i == len(images):
            print(f"  pose {i}/{len(images)} ...")
        sample = analyze_image(path, landmarker)
        if sample is None:
            skipped["no_face_or_pose"] += 1
            continue
        if abs(sample.roll) > args.max_roll:
            skipped["roll"] += 1
            continue
        if not in_range(sample.yaw, args.yaw_min, args.yaw_max, args.step):
            skipped["yaw_range"] += 1
            continue
        if not in_range(sample.pitch, args.pitch_min, args.pitch_max, args.step):
            skipped["pitch_range"] += 1
            continue
        accepted.append(sample)

    bins: dict[tuple[float, float], list[FaceSample]] = defaultdict(list)
    for s in accepted:
        key = bin_key(s.yaw, s.pitch, args)
        bins[key].append(s)

    selected: list[tuple[tuple[float, float], FaceSample]] = []
    for key, items in bins.items():
        items_sorted = sorted(items, key=lambda x: x.score, reverse=True)
        keep = items_sorted[: max(1, args.per_bin)]
        # Replace bin contents with kept only (for coverage counts)
        bins[key] = keep
        for s in keep:
            selected.append((key, s))

    # Empty bins intentionally absent from bins dict until we report full grid
    filled_bins = {k: v for k, v in bins.items() if v}

    print(f"Pose OK: {len(accepted)} | selected: {len(selected)} | skip: {dict(skipped)}")

    if args.dry_run:
        # Still write reports next to a suggested output for inspection
        report_dir = output_dir
        report_dir.mkdir(parents=True, exist_ok=True)
    else:
        report_dir = output_dir

    manifest_path = report_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "out_path",
                "source_path",
                "yaw",
                "pitch",
                "roll",
                "yaw_bin",
                "pitch_bin",
                "sharpness",
                "face_frac",
                "score",
                "kind",
            ]
        )
        for (yaw_b, pitch_b), sample in sorted(
            selected, key=lambda t: (t[0][0], t[0][1], -t[1].score)
        ):
            rel_folder = folder_name(yaw_b, pitch_b)
            out_name = sample.path.name
            out_path = report_dir / rel_folder / out_name
            if not args.dry_run:
                place_file(sample.path, out_path, args.copy_mode)
            w.writerow(
                [
                    str(out_path if not args.dry_run else Path(rel_folder) / out_name),
                    str(sample.path),
                    f"{sample.yaw:.3f}",
                    f"{sample.pitch:.3f}",
                    f"{sample.roll:.3f}",
                    f"{yaw_b:.1f}",
                    f"{pitch_b:.1f}",
                    f"{sample.sharpness:.2f}",
                    f"{sample.face_frac:.4f}",
                    f"{sample.score:.2f}",
                    "real",
                ]
            )

    write_coverage_csv(report_dir / "coverage.csv", filled_bins, args)
    try:
        write_heatmap_png(report_dir / "coverage_heatmap.png", filled_bins, args)
    except Exception as exc:  # noqa: BLE001
        print(f"Heatmap skipped: {exc}")

    # Summary
    yaw_n = int(round((args.yaw_max - args.yaw_min) / args.step)) + 1
    pitch_n = int(round((args.pitch_max - args.pitch_min) / args.step)) + 1
    total_bins = yaw_n * pitch_n
    filled = len(filled_bins)
    print()
    print(f"Grid: yaw [{args.yaw_min},{args.yaw_max}] pitch [{args.pitch_min},{args.pitch_max}] step {args.step}")
    print(f"Bins filled: {filled}/{total_bins} ({100.0 * filled / total_bins:.1f}%)")
    print(f"Manifest: {manifest_path}")
    print(f"Coverage: {report_dir / 'coverage.csv'}")
    if args.dry_run:
        print("Dry run: no image copies written.")
    else:
        print(f"Dataset root: {report_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
