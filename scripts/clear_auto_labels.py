#!/usr/bin/env python3
"""Remove incorrect auto-generated labels and extracted frames."""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PATHS = [
    ROOT / "data" / "dataset" / "staging" / "labels",
    ROOT / "data" / "dataset" / "staging" / "preview",
    ROOT / "data" / "dataset" / "staging" / "images",
    ROOT / "data" / "dataset" / "images" / "train",
    ROOT / "data" / "dataset" / "images" / "val",
    ROOT / "data" / "dataset" / "labels" / "train",
    ROOT / "data" / "dataset" / "labels" / "val",
    ROOT / "data" / "web_jobs",
]

FILES = [
    ROOT / "data" / "dataset" / "label_manifest.json",
    ROOT / "data" / "dataset" / "segments.json",
]


def main() -> None:
    for path in PATHS:
        if not path.exists():
            continue
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
        print(f"cleared {path}")

    for file in FILES:
        if file.exists():
            file.unlink()
            print(f"removed {file}")

    print("Done. Use the web annotator for manual polygon labels.")


if __name__ == "__main__":
    main()
