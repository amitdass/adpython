#!/usr/bin/env python3
"""
Extract sharp face frames from a local video, favoring side / three-quarter poses.

No uploads. Reads a local .mp4, writes selected frames to an output folder.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from curate_real_only import (
    MODEL_URL,
    analyze_image,
    build_landmarker,
    ensure_model,
    sharpness_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract side-pose face frames from a local video (no uploads)."
    )
    p.add_argument("--video", type=Path, required=True, help="Path to .mp4 / video file")
    p.add_argument("--output", type=Path, required=True, help="Output folder for frames")
    p.add_argument(
        "--every-n",
        type=int,
        default=3,
        help="Sample every Nth frame before pose/sharpness filters (default 3).",
    )
    p.add_argument(
        "--min-yaw",
        type=float,
        default=22.0,
        help="Keep frames with |yaw| >= this (degrees). Side / three-quarter.",
    )
    p.add_argument(
        "--max-yaw",
        type=float,
        default=85.0,
        help="Ignore extreme/broken estimates above this |yaw|.",
    )
    p.add_argument(
        "--max-pitch",
        type=float,
        default=30.0,
        help="Skip frames with |pitch| above this.",
    )
    p.add_argument(
        "--max-roll",
        type=float,
        default=25.0,
        help="Skip frames with |roll| above this.",
    )
    p.add_argument(
        "--min-sharpness",
        type=float,
        default=40.0,
        help="Minimum Laplacian variance (blur filter). Lower = more frames.",
    )
    p.add_argument(
        "--yaw-bin",
        type=float,
        default=5.0,
        help="Keep best sharp frame per |signed| yaw bin of this size.",
    )
    p.add_argument(
        "--also-frontal",
        action="store_true",
        help="Also keep a few sharp near-frontal frames (|yaw| < 12) for reference.",
    )
    p.add_argument("--model", type=Path, default=None)
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report counts only; write no images.",
    )
    return p.parse_args()


def yaw_bin(yaw: float, step: float) -> float:
    return round(yaw / step) * step


def main() -> int:
    args = parse_args()
    video_path = args.video.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not video_path.is_file():
        print(f"Video not found: {video_path}", file=sys.stderr)
        return 1

    model_path = args.model or (
        Path(__file__).resolve().parent / "models" / "face_landmarker.task"
    )
    try:
        ensure_model(model_path)
    except Exception as exc:  # noqa: BLE001
        print(
            f"Could not fetch model ({exc}).\n"
            f"Download manually and pass --model:\n  {MODEL_URL}",
            file=sys.stderr,
        )
        return 1

    landmarker = build_landmarker(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {video_path}")
    print(f"Frames≈{total}  fps≈{fps:.2f}  sample every {args.every_n}")

    tmp_dir = output_dir / "_tmp_frames"
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    # Collect candidates: bin_key -> best (sharpness, yaw, pitch, roll, frame_idx, path)
    side_bins: dict[float, tuple] = {}
    frontal_bins: dict[float, tuple] = {}
    stats = defaultdict(int)

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.every_n != 0:
            frame_idx += 1
            continue

        if frame_idx % (args.every_n * 50) == 0:
            print(f"  scanned frame {frame_idx}/{total or '?'} ...")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = sharpness_score(gray)
        if sharp < args.min_sharpness:
            stats["blurry"] += 1
            frame_idx += 1
            continue

        # Write temp JPEG for analyze_image (expects a path)
        if args.dry_run:
            tmp_path = Path(f"/tmp/_pose_probe_{frame_idx}.jpg")
        else:
            tmp_path = tmp_dir / f"f_{frame_idx:06d}.jpg"
        cv2.imwrite(str(tmp_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        sample = analyze_image(tmp_path, landmarker)
        if sample is None:
            stats["no_face"] += 1
            if args.dry_run:
                tmp_path.unlink(missing_ok=True)
            else:
                tmp_path.unlink(missing_ok=True)
            frame_idx += 1
            continue

        yaw, pitch, roll = sample.yaw, sample.pitch, sample.roll
        if abs(roll) > args.max_roll or abs(pitch) > args.max_pitch:
            stats["pose_reject"] += 1
            tmp_path.unlink(missing_ok=True)
            frame_idx += 1
            continue
        if abs(yaw) > args.max_yaw:
            stats["yaw_extreme"] += 1
            tmp_path.unlink(missing_ok=True)
            frame_idx += 1
            continue

        abs_yaw = abs(yaw)
        if abs_yaw >= args.min_yaw:
            key = yaw_bin(yaw, args.yaw_bin)
            prev = side_bins.get(key)
            cand = (sharp, yaw, pitch, roll, frame_idx, tmp_path)
            if prev is None or sharp > prev[0]:
                if prev is not None and not args.dry_run:
                    prev[5].unlink(missing_ok=True)
                side_bins[key] = cand
            else:
                tmp_path.unlink(missing_ok=True)
            stats["side_seen"] += 1
        elif args.also_frontal and abs_yaw < 12.0:
            key = yaw_bin(yaw, args.yaw_bin)
            prev = frontal_bins.get(key)
            cand = (sharp, yaw, pitch, roll, frame_idx, tmp_path)
            if prev is None or sharp > prev[0]:
                if prev is not None and not args.dry_run:
                    prev[5].unlink(missing_ok=True)
                frontal_bins[key] = cand
            else:
                tmp_path.unlink(missing_ok=True)
            stats["frontal_seen"] += 1
        else:
            stats["yaw_too_small"] += 1
            tmp_path.unlink(missing_ok=True)

        frame_idx += 1

    cap.release()

    print()
    print(
        f"Side bins kept: {len(side_bins)} | "
        f"frontal bins kept: {len(frontal_bins)} | stats: {dict(stats)}"
    )

    if args.dry_run:
        print("Dry run — no output written. Yaw bins that would be kept:")
        for key in sorted(side_bins.keys()):
            sharp, yaw, pitch, roll, fi, _ = side_bins[key]
            t = fi / fps if fps > 1e-6 else fi
            print(
                f"  yaw_bin={key:+.0f}  yaw={yaw:+.1f} pitch={pitch:+.1f} "
                f"sharp={sharp:.0f} frame={fi} t≈{t:.1f}s"
            )
        return 0

    side_dir = output_dir / "side"
    front_dir = output_dir / "frontal_ref"
    side_dir.mkdir(parents=True, exist_ok=True)
    if args.also_frontal:
        front_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.csv"
    rows: list[list[object]] = []

    def export_bin(
        bins: dict[float, tuple], out_root: Path, kind: str
    ) -> None:
        for key in sorted(bins.keys()):
            sharp, yaw, pitch, roll, fi, tmp_path = bins[key]
            t = fi / fps if fps > 1e-6 else 0.0
            name = (
                f"{kind}_yaw{yaw:+06.1f}_pitch{pitch:+05.1f}_"
                f"f{fi:06d}_t{t:06.1f}s.jpg"
            )
            dest = out_root / name
            if tmp_path.is_file():
                tmp_path.replace(dest)
            rows.append(
                [
                    str(dest),
                    str(video_path),
                    f"{yaw:.3f}",
                    f"{pitch:.3f}",
                    f"{roll:.3f}",
                    f"{sharp:.2f}",
                    fi,
                    f"{t:.3f}",
                    kind,
                ]
            )

    export_bin(side_bins, side_dir, "side")
    if args.also_frontal:
        export_bin(frontal_bins, front_dir, "frontal")

    # Cleanup leftover temps
    if tmp_dir.is_dir():
        for p in tmp_dir.glob("*"):
            p.unlink(missing_ok=True)
        tmp_dir.rmdir()

    with manifest_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "out_path",
                "video",
                "yaw",
                "pitch",
                "roll",
                "sharpness",
                "frame_index",
                "time_sec",
                "kind",
            ]
        )
        w.writerows(rows)

    print(f"Wrote {len(rows)} images → {output_dir}")
    print(f"  side/         = side & three-quarter poses (|yaw|≥{args.min_yaw})")
    if args.also_frontal:
        print("  frontal_ref/  = a few near-frontal references")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
