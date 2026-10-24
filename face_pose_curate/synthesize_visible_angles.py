#!/usr/bin/env python3
"""
Generate *visibly different* face angles from local source photos (no uploads).

Unlike the mild gap-fill pass, this:
  - scans a whole tree for more sources (e.g. all of wcc1bex)
  - skips tiny pose deltas (near-duplicates)
  - applies amplified landmark rotation so turns read clearly
  - fans each chosen source into multiple target yaw/pitch views

Output: output/real_sources/ + output/synth_angles/
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
    FaceSample,
    MODEL_URL,
    analyze_image,
    build_landmarker,
    ensure_model,
    folder_name,
    iter_images,
    place_file,
)

# Skip prior outputs / junk when scanning a wide tree
EXCLUDE_DIR_NAMES = {
    "saved",
    "saved_diverse",
    "saved_visible",
    "synth",
    "real",
    "real_sources",
    "synth_angles",
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "models",
    "node_modules",
    ".thumbnails",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Synthesize visibly different face angles (local only)."
    )
    p.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Root folder to scan recursively for face images.",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output root for real_sources/ + synth_angles/.",
    )
    p.add_argument("--max-roll", type=float, default=DEFAULT_MAX_ROLL)
    p.add_argument(
        "--max-sources",
        type=int,
        default=120,
        help="How many diverse real sources to fan out from.",
    )
    p.add_argument(
        "--amplify",
        type=float,
        default=2.0,
        help="Multiply applied yaw/pitch rotation (makes turns more visible).",
    )
    p.add_argument(
        "--min-yaw-delta",
        type=float,
        default=12.0,
        help="Do not write a synth closer than this yaw delta to its source.",
    )
    p.add_argument(
        "--yaw-offsets",
        type=str,
        default="-45,-30,-20,20,30,45",
        help="Comma-separated yaw offsets (deg) added to each source pose.",
    )
    p.add_argument(
        "--pitch-offsets",
        type=str,
        default="-12,0,12",
        help="Comma-separated pitch offsets (deg) added to each source pose.",
    )
    p.add_argument("--model", type=Path, default=None)
    p.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
    )
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def parse_float_list(s: str) -> list[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def collect_exclude_dirs(input_dir: Path, output_dir: Path) -> set[Path]:
    exclude = {output_dir.resolve()}
    for path in input_dir.rglob("*"):
        if path.is_dir() and path.name in EXCLUDE_DIR_NAMES:
            exclude.add(path.resolve())
    return exclude


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


def warp_face_visible(
    bgr: np.ndarray,
    landmarks,
    d_yaw: float,
    d_pitch: float,
    amplify: float,
) -> np.ndarray | None:
    """Stronger triangle warp for visibly different head turns."""
    h, w = bgr.shape[:2]
    pts = landmarks_to_xyz(landmarks, w, h)
    center = pts.mean(axis=0)

    # Amplify rotation so aligned crops still show a clear turn
    r = rot_x(d_pitch * amplify) @ rot_y(d_yaw * amplify)
    dst_pts = (pts - center) @ r.T + center

    # Extra horizontal shear on x from yaw for clearer cheek turn
    shear = np.tan(math.radians(d_yaw * amplify * 0.25))
    dst_pts = dst_pts.copy()
    dst_pts[:, 0] += (dst_pts[:, 2] - center[2]) * shear

    src2 = pts[:, :2]
    dst2 = dst_pts[:, :2]
    pad = 4.0
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

    hull = cv2.convexHull(dst2.astype(np.float32))
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillConvexPoly(face_mask, np.int32(hull), 1)
    missing = (face_mask > 0) & (weight < 0.5)
    if np.any(missing):
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

    outside = face_mask == 0
    if np.any(outside):
        blur = cv2.GaussianBlur(bgr, (41, 41), 0)
        out[outside] = (blur[outside] * 0.25).astype(np.uint8)

    if float((weight > 0.5).mean()) < 0.05:
        return None
    return out


def pick_diverse_sources(pool: list[FaceSample], max_sources: int) -> list[FaceSample]:
    """Greedy farthest-point sampling in yaw/pitch, preferring sharp faces."""
    if len(pool) <= max_sources:
        return sorted(pool, key=lambda s: s.score, reverse=True)

    remaining = sorted(pool, key=lambda s: s.score, reverse=True)
    chosen = [remaining.pop(0)]
    while len(chosen) < max_sources and remaining:
        best_i = 0
        best_d = -1.0
        for i, cand in enumerate(remaining):
            d = min(
                math.hypot(cand.yaw - c.yaw, cand.pitch - c.pitch) for c in chosen
            )
            # Prefer farther + reasonably sharp
            d = d + 0.001 * math.log1p(cand.score)
            if d > best_d:
                best_d = d
                best_i = i
        chosen.append(remaining.pop(best_i))
    return chosen


def main() -> int:
    args = parse_args()
    input_dir = args.input.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    yaw_offsets = parse_float_list(args.yaw_offsets)
    pitch_offsets = parse_float_list(args.pitch_offsets)

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
    exclude = collect_exclude_dirs(input_dir, output_dir)

    images = list(iter_images(input_dir, exclude))
    print(f"Found {len(images)} images under {input_dir}")
    print(f"Excluded dirs named: {sorted(EXCLUDE_DIR_NAMES)}")

    pool: list[FaceSample] = []
    skipped = defaultdict(int)
    for i, path in enumerate(images, 1):
        if i % 100 == 0 or i == len(images):
            print(f"  pose {i}/{len(images)} ...")
        sample = analyze_image(path, landmarker)
        if sample is None:
            skipped["no_face_or_pose"] += 1
            continue
        if abs(sample.roll) > args.max_roll:
            skipped["roll"] += 1
            continue
        pool.append(sample)

    print(f"Usable faces: {len(pool)} | skip: {dict(skipped)}")
    if not pool:
        return 1

    sources = pick_diverse_sources(pool, args.max_sources)
    print(f"Selected {len(sources)} diverse sources to fan out")

    jobs: list[tuple[FaceSample, float, float, float, float]] = []
    for src in sources:
        for dy in yaw_offsets:
            for dp in pitch_offsets:
                if abs(dy) < args.min_yaw_delta and abs(dp) < 8.0:
                    continue  # skip near-duplicate views
                if abs(dy) < args.min_yaw_delta and dp == 0.0:
                    continue
                tgt_yaw = src.yaw + dy
                tgt_pitch = src.pitch + dp
                jobs.append((src, tgt_yaw, tgt_pitch, dy, dp))

    print(f"Synth jobs (visible deltas only): {len(jobs)}")
    if args.dry_run:
        print("Dry run — listing first 20 planned targets:")
        for src, ty, tp, dy, dp in jobs[:20]:
            print(
                f"  {src.path.name}: yaw {src.yaw:+.1f}->{ty:+.1f} "
                f"(Δ{dy:+.0f}) pitch {src.pitch:+.1f}->{tp:+.1f} (Δ{dp:+.0f})"
            )
        print(f"Total planned: {len(jobs)}")
        return 0

    # Copy chosen sources for reference
    for src in sources:
        rel = folder_name(
            round(src.yaw / 5.0) * 5.0,
            round(src.pitch / 5.0) * 5.0,
        )
        out_path = output_dir / "real_sources" / rel / src.path.name
        place_file(src.path, out_path, args.copy_mode)

    manifest_rows: list[list[object]] = []
    for src in sources:
        manifest_rows.append(
            [
                str(src.path),
                str(src.path),
                f"{src.yaw:.3f}",
                f"{src.pitch:.3f}",
                f"{src.roll:.3f}",
                "real_source",
                "0",
                "0",
            ]
        )

    ok = 0
    fail = defaultdict(int)
    # Cache landmarks per source path
    landmark_cache: dict[Path, object] = {}
    image_cache: dict[Path, np.ndarray] = {}

    for ji, (src, tgt_yaw, tgt_pitch, dy, dp) in enumerate(jobs, 1):
        if ji % 25 == 0 or ji == len(jobs):
            print(f"  synth {ji}/{len(jobs)} (ok={ok}) ...")

        if src.path not in image_cache:
            bgr = cv2.imread(str(src.path), cv2.IMREAD_COLOR)
            if bgr is None:
                fail["read"] += 1
                continue
            image_cache[src.path] = bgr
            landmark_cache[src.path] = detect_landmarks(landmarker, bgr)
        bgr = image_cache[src.path]
        lms = landmark_cache[src.path]
        if lms is None:
            fail["landmarks"] += 1
            continue

        warped = warp_face_visible(bgr, lms, dy, dp, args.amplify)
        if warped is None:
            fail["warp"] += 1
            continue

        # Soft accept: must still detect a face (identity texture is from source)
        tmp = output_dir / "_tmp_vis.jpg"
        cv2.imwrite(str(tmp), warped, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        checked = analyze_image(tmp, landmarker)
        if checked is None:
            fail["undetected"] += 1
            tmp.unlink(missing_ok=True)
            continue

        rel = folder_name(
            round(tgt_yaw / 5.0) * 5.0,
            round(tgt_pitch / 5.0) * 5.0,
        )
        out_name = (
            f"vis_y{dy:+.0f}_p{dp:+.0f}_from_{src.path.stem}.jpg"
        )
        out_path = output_dir / "synth_angles" / rel / out_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.unlink(missing_ok=True)
        shutil.move(str(tmp), str(out_path))
        ok += 1
        manifest_rows.append(
            [
                str(out_path),
                str(src.path),
                f"{checked.yaw:.3f}",
                f"{checked.pitch:.3f}",
                f"{checked.roll:.3f}",
                "synth_visible",
                f"{dy:.1f}",
                f"{dp:.1f}",
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
                "kind",
                "delta_yaw",
                "delta_pitch",
            ]
        )
        w.writerows(manifest_rows)

    print()
    print(f"Sources used: {len(sources)}")
    print(f"New angle images written: {ok}")
    print(f"Failed: {dict(fail)}")
    print(f"Output: {output_dir}")
    print(f"  real_sources/  = originals you fanned from")
    print(f"  synth_angles/  = NEW visibly turned faces")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
