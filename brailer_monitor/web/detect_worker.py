"""Run a single video detection in an isolated subprocess.

A CUDA "illegal memory access" (cudaErrorIllegalAddress) permanently corrupts
the CUDA context of the *whole* process: every later CUDA call in that process
keeps failing. The web server runs detection in background threads, so such an
error would otherwise wedge the server until it is restarted manually.

To stay recoverable, the pipeline launches this module as a separate process
for each detection job. If the GPU dies here, only this process dies; the server
keeps a clean CUDA context and the next job simply spawns a fresh worker.

Progress is reported by atomically writing a small JSON file that the parent
polls. Cancellation is handled by the parent terminating this process.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _write_progress(progress_file: Path, processed: int, total: int, with_det: int) -> None:
    tmp = progress_file.with_suffix(progress_file.suffix + ".tmp")
    try:
        tmp.write_text(
            json.dumps({"processed": processed, "total": total, "with": with_det}),
            encoding="utf-8",
        )
        os.replace(tmp, progress_file)
    except Exception:
        # Progress reporting is best-effort; never let it crash detection.
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated single-video detection worker")
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--device", default="0")
    parser.add_argument("--segmentation", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--progress-file", required=True)
    args = parser.parse_args(argv)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_file = output_dir / "worker_error.txt"
    progress_file = Path(args.progress_file)

    device: str | int = args.device
    if isinstance(device, str) and device.lstrip("-").isdigit():
        device = int(device)

    use_segmentation = None if args.segmentation == "auto" else args.segmentation == "yes"

    try:
        from ..video_detect import detect_video

        def on_progress(processed: int, total: int, with_det: int) -> None:
            _write_progress(progress_file, processed, total, with_det)

        detect_video(
            Path(args.video),
            Path(args.model),
            output_dir=output_dir,
            frame_stride=args.frame_stride,
            confidence=args.confidence,
            device=device,
            use_segmentation=use_segmentation,
            on_progress=on_progress,
        )
        return 0
    except Exception as exc:  # noqa: BLE001 - report any failure to the parent
        message = f"{type(exc).__name__}: {exc}"
        try:
            error_file.write_text(
                message + "\n\n" + traceback.format_exc(),
                encoding="utf-8",
            )
        except Exception:
            pass
        print(message, file=sys.stderr)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
