#!/usr/bin/env python3
"""End-to-end: extract brailer frames -> SAM label -> train/val split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from brailer_monitor.frame_extractor import ExtractOptions, extract_brailer_frames, save_segment_manifest, scan_brailer_segments
from brailer_monitor.labeling import label_extracted_frames, split_train_val, update_dataset_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and label brailer_loaded from video")
    parser.add_argument("--video", default="data/raw/JJR-102283_stream04_260310_202016.mp4")
    parser.add_argument("--scan-stride", type=int, default=15)
    parser.add_argument("--extract-stride", type=int, default=15)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sam-model", default="models/mobile_sam.pt")
    parser.add_argument("--prefix", default="lake_win")
    parser.add_argument("--skip-label", action="store_true", help="Only extract frames, skip SAM labeling")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    video_path = root / args.video
    dataset_root = root / "data" / "dataset"
    staging_img = dataset_root / "staging" / "images"
    staging_lbl = dataset_root / "staging" / "labels"
    preview_dir = dataset_root / "staging" / "preview"

    opts = ExtractOptions(scan_stride=args.scan_stride, extract_stride=args.extract_stride)

    segments, fps, _ = scan_brailer_segments(video_path, opts)
    print(f"Found {len(segments)} brailer segments")
    for seg in segments:
        print(f"  #{seg.segment_id}: {seg.start_sec:.1f}s - {seg.end_sec:.1f}s")

    extracted, segments = extract_brailer_frames(
        video_path,
        staging_img,
        prefix=args.prefix,
        options=opts,
        segments=segments,
        preview_dir=preview_dir,
    )
    save_segment_manifest(
        segments,
        extracted,
        dataset_root / "segments.json",
        video_path=video_path,
        fps=fps,
    )
    print(f"Extracted {len(extracted)} frames")

    if args.skip_label:
        return

    from ultralytics import SAM

    sam_weights = root / args.sam_model
    sam_model = SAM(str(sam_weights))
    records = label_extracted_frames(extracted, staging_lbl, sam_model, preview_dir=preview_dir)

    manifest_path = dataset_root / "label_manifest.json"
    manifest_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    train_n, val_n = split_train_val(staging_img, staging_lbl, dataset_root, args.val_ratio, args.seed)
    update_dataset_yaml(root, train_n, val_n)
    print(f"Train: {train_n}, Val: {val_n}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
