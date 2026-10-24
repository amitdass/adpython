#!/usr/bin/env python3
"""
Local pose gap-fill for aligned faces (no photo uploads).

For empty yaw/pitch bins, synthesize a new view by 3D-rotating MediaPipe
face landmarks and triangle-warping texture from the nearest real source
image in the aligned folder.

This creates *new* images (warped), not copies. Quality drops as the angle
delta from the source grows; extremes beyond --max-delta-* stay empty.
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import Delaunay

from curate_real_only import (
    DEFAULT_MAX_ROLL,
    DEFAULT_PITCH_MAX,
    DEFAULT_PITCH_MIN,
    DEFAULT_STEP,
    DEFAULT_YAW_MAX,
    DEFAULT_YAW_MIN,
    FaceSample,
    MODEL_URL,
    analyze_image,
    bin_center,
    build_landmarker,
    ensure_model,
    folder_name,
    in_range,
    iter_images,
    place_file,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesize missing yaw/pitch views from aligned faces (local)."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Aligned face folder (all images used as synthesis sources).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset root (real copies + synth images).",
    )
    p.add_argument("--yaw-min", type=float, default=DEFAULT_YAW_MIN)
    p.add_argument("--yaw-max", type=float, default=DEFAULT_YAW_MAX)
    p.add_argument("--pitch-min", type=float, default=DEFAULT_PITCH_MIN)
    p.add_argument("--pitch-max", type=float, default=DEFAULT_PITCH_MAX)
    p.add_argument("--step", type=float, default=DEFAULT_STEP)
    p.add_argument("--max-roll", type=float, default=DEFAULT_MAX_ROLL)
    p.add_argument(
        "--max-delta-yaw",
        type=float,
        default=20.0,
        help="Max yaw degrees to synthesize away from a real source.",
    )
    p.add_argument(
        "--max-delta-pitch",
        type=float,
        default=12.0,
        help="Max pitch degrees to synthesize away from a real source.",
    )
    p.add_argument(
        "--per-bin-real",
        type=int,
        default=1,
        help="How many real images to keep per occupied bin.",
    )
    p.add_argument("--model", type=Path, default=None)
    p.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to place REAL images into output (synths are always written).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan synths and write reports only; no image files.",
    )
    return p.parse_args()


def grid_axes(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    yaws: list[float] = []
    v = args.yaw_min
    while v <= args.yaw_max + 1e-9:
        yaws.append(round(v, 6))
        v += args.step
    pitches: list[float] = []
    v = args.pitch_min
    while v <= args.pitch_max + 1e-9:
        pitches.append(round(v, 6))
        v += args.step
    return yaws, pitches


def rot_x(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def landmarks_to_xyz(landmarks, width: int, height: int) -> np.ndarray:
    return np.array(
        [[lm.x * width, lm.y * height, lm.z * width] for lm in landmarks],
        dtype=np.float64,
    )


def detect_landmarks(landmarker, bgr: np.ndarray):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def warp_face_to_pose(
    bgr: np.ndarray,
    landmarks,
    src_yaw: float,
    src_pitch: float,
    tgt_yaw: float,
    tgt_pitch: float,
) -> np.ndarray | None:
    """Triangle-warp face texture after rotating landmark xyz toward target pose."""
    h, w = bgr.shape[:2]
    pts = landmarks_to_xyz(landmarks, w, h)
    center = pts.mean(axis=0)

    d_yaw = tgt_yaw - src_yaw
    d_pitch = tgt_pitch - src_pitch
    # Apply inverse of source pose then target-ish delta in camera space:
    # rotate by (d_pitch, d_yaw) around face center.
    r = rot_x(d_pitch) @ rot_y(d_yaw)
    dst_pts = (pts - center) @ r.T + center

    src2 = pts[:, :2]
    dst2 = dst_pts[:, :2]

    # Keep points inside a padded image so edge triangles still warp
    pad = 2.0
    src2 = np.clip(src2, -pad, [w - 1 + pad, h - 1 + pad])
    dst2 = np.clip(dst2, -pad, [w - 1 + pad, h - 1 + pad])

    try:
        tri = Delaunay(src2)
    except Exception:  # noqa: BLE001
        return None

    out = np.zeros_like(bgr)
    weight = np.zeros((h, w), dtype=np.float32)

    for simplex in tri.simplices:
        s_tri = src2[simplex].astype(np.float32)
        d_tri = dst2[simplex].astype(np.float32)
        # Skip degenerate triangles
        if abs(cv2.contourArea(s_tri)) < 1.0 or abs(cv2.contourArea(d_tri)) < 1.0:
            continue
        x, y, bw, bh = cv2.boundingRect(d_tri)
        x2, y2 = x + bw, y + bh
        x, y = max(0, x), max(0, y)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x or y2 <= y:
            continue
        try:
            m = cv2.getAffineTransform(s_tri, d_tri)
        except cv2.error:
            continue
        warped = cv2.warpAffine(
            bgr,
            m,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.int32(d_tri), 1)
        roi = mask[y:y2, x:x2].astype(bool)
        if not np.any(roi):
            continue
        out[y:y2, x:x2][roi] = warped[y:y2, x:x2][roi]
        weight[y:y2, x:x2][roi] = 1.0

    # Soft face hull: fill holes from source where warp missed
    hull = cv2.convexHull(dst2.astype(np.float32))
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, np.int32(hull), 1)
    missing = (face_mask > 0) & (weight < 0.5)
    if np.any(missing):
        # Pull from a globally affine-approximated warp as fallback
        m_global, _ = cv2.estimateAffinePartial2D(
            src2.astype(np.float32), dst2.astype(np.float32)
        )
        if m_global is not None:
            approx = cv2.warpAffine(
                bgr,
                m_global,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT101,
            )
            out[missing] = approx[missing]
            weight[missing] = 1.0

    # Outside face: keep a darkened/blurred source so frame isn't empty
    outside = face_mask == 0
    if np.any(outside):
        blur = cv2.GaussianBlur(bgr, (31, 31), 0)
        out[outside] = (blur[outside] * 0.35).astype(np.uint8)

    # Reject near-empty results
    if float(weight.mean()) < 0.08:
        return None
    return out


def pose_distance(a: FaceSample, yaw: float, pitch: float) -> float:
    # Slightly weight yaw (usually the wider axis)
    return math.hypot(1.0 * (a.yaw - yaw), 1.2 * (a.pitch - pitch))


def write_coverage(
    path: Path,
    status: dict[tuple[float, float], str],
    args: argparse.Namespace,
) -> None:
    yaws, pitches = grid_axes(args)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["yaw", "pitch", "status"])
        for pitch in pitches:
            for yaw in yaws:
                w.writerow([yaw, pitch, status.get((yaw, pitch), "empty")])


def write_heatmap(
    path: Path,
    status: dict[tuple[float, float], str],
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt

    yaws, pitches = grid_axes(args)
    # 0 empty, 1 real, 2 synth
    grid = np.zeros((len(pitches), len(yaws)), dtype=np.float64)
    for yi, yaw in enumerate(yaws):
        for pi, pitch in enumerate(pitches):
            s = status.get((yaw, pitch), "empty")
            grid[pi, yi] = {"empty": 0, "real": 1, "synth": 2}.get(s, 0)

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
        vmin=0,
        vmax=2,
        cmap="viridis",
    )
    ax.set_xlabel("Yaw (deg)")
    ax.set_ylabel("Pitch (deg)")
    ax.set_title("Coverage: 0=empty, 1=real, 2=synth")
    fig.colorbar(im, ax=ax, ticks=[0, 1, 2])
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not input_dir.is_dir():
        print(f"Input folder not found: {input_dir}", file=sys.stderr)
        return 1

    model_path = args.model or (
        Path(__file__).resolve().parent / "models" / "face_landmarker.task"
    )
    try:
        ensure_model(model_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not fetch model automatically ({exc}).\n"
            f"Download manually and pass --model PATH:\n  {MODEL_URL}",
            file=sys.stderr,
        )
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    landmarker = build_landmarker(model_path)

    exclude = {output_dir}
    # Also skip common prior output folders under aligned/
    for name in ("saved", "saved_diverse", "saved_with_synth"):
        exclude.add((input_dir / name).resolve())

    images = list(iter_images(input_dir, exclude))
    print(f"Found {len(images)} aligned images under {input_dir}")

    pool: list[FaceSample] = []
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
        pool.append(sample)

    print(f"Source pool: {len(pool)} | skip: {dict(skipped)}")
    if not pool:
        print("No usable source faces.", file=sys.stderr)
        return 1

    # Real best-per-bin
    real_bins: dict[tuple[float, float], list[FaceSample]] = defaultdict(list)
    for s in pool:
        if not in_range(s.yaw, args.yaw_min, args.yaw_max, args.step):
            continue
        if not in_range(s.pitch, args.pitch_min, args.pitch_max, args.step):
            continue
        key = (
            bin_center(s.yaw, args.yaw_min, args.step),
            bin_center(s.pitch, args.pitch_min, args.step),
        )
        real_bins[key].append(s)

    real_selected: dict[tuple[float, float], list[FaceSample]] = {}
    for key, items in real_bins.items():
        keep = sorted(items, key=lambda x: x.score, reverse=True)[
            : max(1, args.per_bin_real)
        ]
        real_selected[key] = keep

    yaws, pitches = grid_axes(args)
    status: dict[tuple[float, float], str] = {}
    manifest_rows: list[list[object]] = []

    # Place reals
    for (yaw_b, pitch_b), samples in sorted(real_selected.items()):
        status[(yaw_b, pitch_b)] = "real"
        for sample in samples:
            rel = folder_name(yaw_b, pitch_b)
            out_path = output_dir / "real" / rel / sample.path.name
            if not args.dry_run:
                place_file(sample.path, out_path, args.copy_mode)
            manifest_rows.append(
                [
                    str(out_path),
                    str(sample.path),
                    f"{sample.yaw:.3f}",
                    f"{sample.pitch:.3f}",
                    f"{sample.roll:.3f}",
                    f"{yaw_b:.1f}",
                    f"{pitch_b:.1f}",
                    "real",
                    "",
                    "",
                ]
            )

    # Synthesize empty bins
    synth_ok = 0
    synth_skip = defaultdict(int)
    empty_targets = [
        (yaw, pitch)
        for pitch in pitches
        for yaw in yaws
        if (yaw, pitch) not in real_selected
    ]
    print(f"Empty bins to try: {len(empty_targets)}")

    for ti, (yaw_t, pitch_t) in enumerate(empty_targets, 1):
        if ti % 25 == 0 or ti == len(empty_targets):
            print(f"  synth plan {ti}/{len(empty_targets)} ...")

        # Nearest real source within delta limits
        candidates = []
        for s in pool:
            dy = abs(s.yaw - yaw_t)
            dp = abs(s.pitch - pitch_t)
            if dy <= args.max_delta_yaw and dp <= args.max_delta_pitch:
                candidates.append((pose_distance(s, yaw_t, pitch_t), s))
        if not candidates:
            synth_skip["no_source_in_delta"] += 1
            status[(yaw_t, pitch_t)] = "empty"
            continue

        _dist, src = min(candidates, key=lambda t: t[0])
        if args.dry_run:
            status[(yaw_t, pitch_t)] = "synth"
            synth_ok += 1
            manifest_rows.append(
                [
                    str(
                        output_dir
                        / "synth"
                        / folder_name(yaw_t, pitch_t)
                        / f"from_{src.path.stem}.jpg"
                    ),
                    str(src.path),
                    f"{yaw_t:.3f}",
                    f"{pitch_t:.3f}",
                    "0.000",
                    f"{yaw_t:.1f}",
                    f"{pitch_t:.1f}",
                    "synth",
                    f"{src.yaw:.3f}",
                    f"{src.pitch:.3f}",
                ]
            )
            continue

        bgr = cv2.imread(str(src.path), cv2.IMREAD_COLOR)
        if bgr is None:
            synth_skip["read_fail"] += 1
            status[(yaw_t, pitch_t)] = "empty"
            continue
        lms = detect_landmarks(landmarker, bgr)
        if lms is None:
            synth_skip["landmark_fail"] += 1
            status[(yaw_t, pitch_t)] = "empty"
            continue

        warped = warp_face_to_pose(bgr, lms, src.yaw, src.pitch, yaw_t, pitch_t)
        if warped is None:
            synth_skip["warp_fail"] += 1
            status[(yaw_t, pitch_t)] = "empty"
            continue

        # Verify synthesized pose is closer to target than before (soft check)
        tmp_path = output_dir / "_tmp_synth.jpg"
        cv2.imwrite(str(tmp_path), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        checked = analyze_image(tmp_path, landmarker)
        if checked is None:
            synth_skip["repose_undetected"] += 1
            status[(yaw_t, pitch_t)] = "empty"
            tmp_path.unlink(missing_ok=True)
            continue

        # Accept if moved toward target on at least the dominant axis, or within 12°
        err = pose_distance(checked, yaw_t, pitch_t)
        if err > 12.0 and err > pose_distance(src, yaw_t, pitch_t) * 0.85:
            synth_skip["pose_not_improved"] += 1
            status[(yaw_t, pitch_t)] = "empty"
            tmp_path.unlink(missing_ok=True)
            continue

        rel = folder_name(yaw_t, pitch_t)
        out_name = f"synth_y{yaw_t:+.0f}_p{pitch_t:+.0f}_from_{src.path.stem}.jpg"
        out_path = output_dir / "synth" / rel / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        shutil.move(str(tmp_path), str(out_path))

        status[(yaw_t, pitch_t)] = "synth"
        synth_ok += 1
        manifest_rows.append(
            [
                str(out_path),
                str(src.path),
                f"{checked.yaw:.3f}",
                f"{checked.pitch:.3f}",
                f"{checked.roll:.3f}",
                f"{yaw_t:.1f}",
                f"{pitch_t:.1f}",
                "synth",
                f"{src.yaw:.3f}",
                f"{src.pitch:.3f}",
            ]
        )

    manifest_path = output_dir / "manifest.csv"
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
                "kind",
                "source_yaw",
                "source_pitch",
            ]
        )
        w.writerows(manifest_rows)

    write_coverage(output_dir / "coverage.csv", status, args)
    try:
        write_heatmap(output_dir / "coverage_heatmap.png", status, args)
    except Exception as exc:  # noqa: BLE001
        print(f"Heatmap skipped: {exc}")

    total = len(yaws) * len(pitches)
    n_real = sum(1 for s in status.values() if s == "real")
    n_synth = sum(1 for s in status.values() if s == "synth")
    n_empty = total - n_real - n_synth
    print()
    print(
        f"Bins: real={n_real} synth={n_synth} empty={n_empty} "
        f"total={total} ({100.0 * (n_real + n_synth) / total:.1f}% filled)"
    )
    print(f"Synth written/planned: {synth_ok} | synth skip: {dict(synth_skip)}")
    print(f"Manifest: {manifest_path}")
    if args.dry_run:
        print("Dry run only — no synth/real image files written.")
    else:
        print(f"Dataset root: {output_dir}")
        print(f"  real/  = copies from aligned")
        print(f"  synth/ = newly warped views")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
