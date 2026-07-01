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
import signal
import sys
import traceback
from pathlib import Path


def _is_cuda_oom_error(message: str) -> bool:
    lowered = message.lower()
    return "cuda" in lowered and (
        "out of memory" in lowered
        or "memoryallocation" in lowered
        or "cudaerrormemoryallocation" in lowered
    )


def _is_cpu_device(device: str | int) -> bool:
    return isinstance(device, str) and device.lower() == "cpu"


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


def _append_event(events_file: Path, frame: dict) -> None:
    try:
        with events_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(frame, ensure_ascii=False))
            handle.write("\n")
    except Exception:
        # Incremental timeline reporting is best-effort; the final manifest is
        # still written and merged when the worker completes.
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Isolated single-video detection worker")
    parser.add_argument("--video")
    parser.add_argument("--stream-url")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frame-stride", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.6)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--device", default="0")
    parser.add_argument("--segmentation", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--sam", choices=["yes", "no"], default="no")
    parser.add_argument("--progress-file", required=True)
    parser.add_argument("--events-file")
    parser.add_argument("--stop-file")
    args = parser.parse_args(argv)
    if not args.video and not args.stream_url:
        parser.error("--video or --stream-url is required")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_file = output_dir / "worker_error.txt"
    progress_file = Path(args.progress_file)

    device: str | int = args.device
    if isinstance(device, str) and device.lstrip("-").isdigit():
        device = int(device)

    use_segmentation = None if args.segmentation == "auto" else args.segmentation == "yes"
    use_sam = args.sam == "yes"

    def on_progress(processed: int, total: int, with_det: int) -> None:
        _write_progress(progress_file, processed, total, with_det)

    events_file = Path(args.events_file) if args.events_file else None

    def on_detection(frame: dict) -> None:
        if events_file is not None:
            _append_event(events_file, frame)

    stop_file = Path(args.stop_file) if args.stop_file else None

    def should_cancel() -> bool:
        return stop_file is not None and stop_file.exists()

    def _handle_signal(signum: int, frame: object) -> None:
        if stop_file is not None:
            try:
                stop_file.write_text("stop", encoding="utf-8")
            except Exception:
                pass
            return
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    try:
        from ..video_detect import detect_stream, detect_video

        if args.stream_url:
            detect_stream(
                args.stream_url,
                Path(args.model),
                output_dir=output_dir,
                frame_stride=args.frame_stride,
                confidence=args.confidence,
                imgsz=args.imgsz,
                device=device,
                use_segmentation=use_segmentation,
                use_sam=use_sam,
                on_progress=on_progress,
                on_detection=on_detection,
                should_cancel=should_cancel,
            )
            return 0

        try:
            detect_video(
                Path(args.video),
                Path(args.model),
                output_dir=output_dir,
                frame_stride=args.frame_stride,
                confidence=args.confidence,
                imgsz=args.imgsz,
                device=device,
                use_segmentation=use_segmentation,
                use_sam=use_sam,
                on_progress=on_progress,
                on_detection=on_detection,
                should_cancel=should_cancel,
            )
        except Exception as exc:
            if _is_cpu_device(device) or not _is_cuda_oom_error(str(exc)):
                raise
            print(
                "CUDA out of memory; retrying this video on CPU.",
                file=sys.stderr,
                flush=True,
            )
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            detect_video(
                Path(args.video),
                Path(args.model),
                output_dir=output_dir,
                frame_stride=args.frame_stride,
                confidence=args.confidence,
                imgsz=args.imgsz,
                device="cpu",
                use_segmentation=use_segmentation,
                use_sam=use_sam,
                on_progress=on_progress,
                on_detection=on_detection,
                should_cancel=should_cancel,
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
