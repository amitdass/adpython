#!/usr/bin/env python3
"""
Extract sharp face frames from a local video, favoring the most side-on poses.

No uploads. Strategy:
  1) Score every sampled frame's yaw
  2) Print yaw distribution (so you can see if the video even has turns)
  3) Keep the top-N highest |yaw| sharp frames (relative) — not a hard 22° cut
  4) Also write labeled debug previews so you can verify visually
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from curate_real_only import (
    MODEL_URL,
    build_landmarker,
    ensure_model,
    estimate_pose_from_landmarks_geom,
    estimate_pose_from_transform_matrix,
    sharpness_score,
)


@dataclass
class Hit:
    frame_idx: int
    time_sec: float
    yaw: float
    pitch: float
    roll: float
    sharp: float
    bgr: np.ndarray


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract most side-like face frames from a local video."
    )
    p.add_argument("--video", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--every-n", type=int, default=2, help="Sample every Nth frame")
    p.add_argument(
        "--top-n",
        type=int,
        default=40,
        help="Keep this many sharpest-among-highest-|yaw| frames (most side-like).",
    )
    p.add_argument(
        "--min-yaw",
        type=float,
        default=8.0,
        help="Ignore frames with |yaw| below this before ranking (default 8).",
    )
    p.add_argument("--max-yaw", type=float, default=90.0)
    p.add_argument("--max-pitch", type=float, default=40.0)
    p.add_argument("--max-roll", type=float, default=35.0)
    p.add_argument(
        "--min-sharpness",
        type=float,
        default=25.0,
        help="Blur cutoff (lower keeps more frames).",
    )
    p.add_argument(
        "--debug-every-sec",
        type=float,
        default=1.0,
        help="Write a labeled preview frame about every N seconds.",
    )
    p.add_argument("--model", type=Path, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def pose_from_frame(landmarker, bgr: np.ndarray) -> tuple[float, float, float] | None:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = landmarker.detect(mp_image)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]
    yaw, pitch, roll = estimate_pose_from_landmarks_geom(lms)
    if result.facial_transformation_matrixes:
        matrix_pose = estimate_pose_from_transform_matrix(
            result.facial_transformation_matrixes[0]
        )
        if matrix_pose is not None:
            myaw, mpitch, mroll = matrix_pose
            # Prefer larger |yaw| signal when geometry is muted (common on video)
            if abs(myaw) > abs(yaw):
                yaw = 0.4 * yaw + 0.6 * myaw
            if abs(mpitch) > abs(pitch):
                pitch = 0.4 * pitch + 0.6 * mpitch
            roll = 0.5 * roll + 0.5 * mroll
    return yaw, pitch, roll


def label_frame(bgr: np.ndarray, yaw: float, pitch: float, roll: float, sharp: float) -> np.ndarray:
    out = bgr.copy()
    text = f"yaw={yaw:+.1f} pitch={pitch:+.1f} roll={roll:+.1f} sharp={sharp:.0f}"
    cv2.putText(
        out,
        text,
        (16, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    # Side-ness bar
    bar_w = int(min(abs(yaw), 90.0) / 90.0 * (out.shape[1] - 40))
    cv2.rectangle(out, (20, 50), (20 + bar_w, 70), (0, 200, 255), -1)
    cv2.putText(
        out,
        "side-ness (|yaw|)",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return out


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
            f"Could not fetch model ({exc}).\nPass --model PATH\n  {MODEL_URL}",
            file=sys.stderr,
        )
        return 1

    landmarker = build_landmarker(model_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", file=sys.stderr)
        return 1

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Video: {video_path}")
    print(f"Frames≈{total}  fps≈{fps:.2f}  sample every {args.every_n}")

    hits: list[Hit] = []
    debug_hits: list[Hit] = []
    stats = defaultdict(int)
    last_debug_t = -1e9

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = frame_idx / fps if fps > 1e-6 else float(frame_idx)

        take = frame_idx % args.every_n == 0
        want_debug = (t - last_debug_t) >= args.debug_every_sec
        if not take and not want_debug:
            frame_idx += 1
            continue

        if take and (frame_idx % (args.every_n * 40) == 0):
            print(f"  scanned frame {frame_idx}/{total or '?'} ...")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharp = sharpness_score(gray)
        pose = pose_from_frame(landmarker, frame)

        if pose is None:
            stats["no_face"] += 1
            if want_debug and not args.dry_run:
                # still save occasional no-face debug so user sees content
                pass
            frame_idx += 1
            continue

        yaw, pitch, roll = pose
        stats["face_ok"] += 1

        if want_debug:
            debug_hits.append(Hit(frame_idx, t, yaw, pitch, roll, sharp, frame.copy()))
            last_debug_t = t

        if not take:
            frame_idx += 1
            continue

        if sharp < args.min_sharpness:
            stats["blurry"] += 1
            frame_idx += 1
            continue
        if abs(roll) > args.max_roll or abs(pitch) > args.max_pitch:
            stats["pose_reject"] += 1
            frame_idx += 1
            continue
        if abs(yaw) > args.max_yaw:
            stats["yaw_extreme"] += 1
            frame_idx += 1
            continue
        if abs(yaw) < args.min_yaw:
            stats["yaw_too_small"] += 1
            frame_idx += 1
            continue

        hits.append(Hit(frame_idx, t, yaw, pitch, roll, sharp, frame.copy()))
        frame_idx += 1

    cap.release()

    if not hits and not debug_hits:
        print("No faces detected in sampled frames.")
        print(f"Stats: {dict(stats)}")
        print(
            "Likely: video has no clear face, face too small, or heavy profile "
            "where the detector fails. Open the video and confirm a visible face turn."
        )
        return 1

    if hits:
        yaws = np.array([h.yaw for h in hits], dtype=np.float64)
        abs_yaws = np.abs(yaws)
        print()
        print(
            "Yaw distribution on candidate faces: "
            f"min={yaws.min():+.1f}  p10={np.percentile(yaws, 10):+.1f}  "
            f"p50={np.percentile(yaws, 50):+.1f}  p90={np.percentile(yaws, 90):+.1f}  "
            f"max={yaws.max():+.1f}"
        )
        print(
            f"|yaw| : p50={np.percentile(abs_yaws, 50):.1f}  "
            f"p90={np.percentile(abs_yaws, 90):.1f}  max={abs_yaws.max():.1f}"
        )
        print(f"Candidates: {len(hits)} | stats: {dict(stats)}")

        if abs_yaws.max() < 15:
            print()
            print(
                "NOTE: max |yaw| is low (<15°). This video may be mostly frontal "
                "(or a face-aligned export). Side poses may not exist in the file."
            )
    else:
        print(f"No candidates above min-yaw={args.min_yaw}. Stats: {dict(stats)}")

    # Rank: highest |yaw|, break ties with sharpness
    ranked = sorted(hits, key=lambda h: (abs(h.yaw), h.sharp), reverse=True)
    # Diversity: avoid near-duplicate times (keep >= 0.35s apart when possible)
    selected: list[Hit] = []
    for h in ranked:
        if len(selected) >= args.top_n:
            break
        if any(abs(h.time_sec - s.time_sec) < 0.35 for s in selected):
            continue
        selected.append(h)
    # If too few due to time gap, fill ignoring gap
    if len(selected) < min(args.top_n, len(ranked)):
        for h in ranked:
            if h in selected:
                continue
            selected.append(h)
            if len(selected) >= args.top_n:
                break

    print(f"Selected most-side frames: {len(selected)} (top-n={args.top_n})")
    for h in selected[:10]:
        print(
            f"  t={h.time_sec:6.2f}s  yaw={h.yaw:+6.1f}  pitch={h.pitch:+5.1f}  "
            f"sharp={h.sharp:.0f}"
        )
    if len(selected) > 10:
        print(f"  ... +{len(selected) - 10} more")

    if args.dry_run:
        print("Dry run — nothing written. Re-run without --dry-run to export.")
        return 0

    side_dir = output_dir / "side"
    debug_dir = output_dir / "debug_labeled"
    side_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[object]] = []
    for h in sorted(selected, key=lambda x: x.time_sec):
        name = (
            f"side_yaw{h.yaw:+06.1f}_pitch{h.pitch:+05.1f}_"
            f"t{h.time_sec:07.2f}s_f{h.frame_idx:06d}.jpg"
        )
        dest = side_dir / name
        cv2.imwrite(str(dest), h.bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        rows.append(
            [
                str(dest),
                str(video_path),
                f"{h.yaw:.3f}",
                f"{h.pitch:.3f}",
                f"{h.roll:.3f}",
                f"{h.sharp:.2f}",
                h.frame_idx,
                f"{h.time_sec:.3f}",
                "side_topn",
            ]
        )

    # Labeled timeline previews (so you can SEE if any side exists)
    for i, h in enumerate(debug_hits):
        labeled = label_frame(h.bgr, h.yaw, h.pitch, h.roll, h.sharp)
        dest = debug_dir / f"dbg_t{h.time_sec:07.2f}s_yaw{h.yaw:+06.1f}.jpg"
        cv2.imwrite(str(dest), labeled, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="") as f:
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

    print()
    print(f"Wrote {len(rows)} side candidates → {side_dir}")
    print(f"Wrote {len(debug_hits)} labeled previews → {debug_dir}")
    print("Open debug first (yaw printed on each frame):")
    print(f'  open "{debug_dir}"')
    print(f'  open "{side_dir}"')
    if hits and float(np.max(np.abs([h.yaw for h in hits]))) < 15:
        print(
            "If debug frames all look frontal, this MP4 likely has no real side poses."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
