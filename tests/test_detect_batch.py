"""Tests for detection batch state transitions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from brailer_monitor.web.detect_pipeline import (
    DetectJob,
    DetectPipelineManager,
    PipelineState,
    _detect_is_active,
    _is_cpu_device,
    _is_cuda_oom_error,
)
from brailer_monitor.web.detect_worker import (
    _is_cpu_device as _worker_is_cpu_device,
    _is_cuda_oom_error as _worker_is_cuda_oom_error,
)


class DetectBatchTests(unittest.TestCase):
    def test_finished_detect_thread_does_not_count_as_active(self) -> None:
        state = PipelineState(
            detect_status="completed",
            detect_queue_pending=0,
            detect_batch_total=2,
            detect_batch_done=2,
        )
        self.assertFalse(_detect_is_active(state, threads={}))

        import threading

        thread = threading.Thread(target=lambda: None)
        thread.start()
        thread.join()
        self.assertFalse(_detect_is_active(state, threads={"done": thread}))

    def test_cuda_oom_error_detection(self) -> None:
        self.assertTrue(_is_cuda_oom_error("CUDA error: out of memory"))
        self.assertTrue(_is_cuda_oom_error("cudaErrorMemoryAllocation"))
        self.assertFalse(_is_cuda_oom_error("CUDA error: illegal memory access"))
        self.assertFalse(_is_cuda_oom_error("File not found"))
        self.assertTrue(_is_cpu_device("cpu"))
        self.assertFalse(_is_cpu_device("0"))
        self.assertFalse(_is_cpu_device(0))
        self.assertTrue(_worker_is_cuda_oom_error("CUDA error: out of memory"))
        self.assertTrue(_worker_is_cuda_oom_error("cudaErrorMemoryAllocation"))
        self.assertFalse(_worker_is_cuda_oom_error("CUDA error: illegal memory access"))
        self.assertTrue(_worker_is_cpu_device("cpu"))
        self.assertFalse(_worker_is_cpu_device(0))

    def test_model_frames_do_not_fall_back_to_current_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = DetectPipelineManager(
                root=root / "pipeline",
                dataset_root=root / "dataset",
                config_dir=root / "config",
            )
            weights = root / "weights.pt"
            weights.write_bytes(b"weights")
            record = manager.model_library.register(
                weights,
                task_type="segment",
                epochs=3,
                class_names=["win"],
                train_images=10,
                val_images=2,
                dataset_frames=[],
            )

            result = manager.model_frames(record.id)

            self.assertEqual(result["source"], "model")
            self.assertEqual(result["total"], 0)
            self.assertEqual(result["frames"], [])

    def test_start_detection_batch_assigns_explicit_video_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_dir = root / "config"
            model_path = root / "model.pt"
            model_path.write_bytes(b"weights")
            videos = []
            for name in ("first.mp4", "second.mp4", "third.mp4"):
                path = root / name
                path.write_bytes(b"video")
                videos.append({"video_path": path, "video_name": name})

            manager = DetectPipelineManager(
                root=root / "pipeline",
                dataset_root=root / "dataset",
                config_dir=config_dir,
            )
            launched: list[tuple[str, int]] = []

            def fake_launch(item):
                launched.append((item["video_name"], item["batch_index"]))
                job = DetectJob(
                    job_id="first-job",
                    video_name=item["video_name"],
                    created_at="2026-06-29T00:00:00+00:00",
                    status="running",
                )
                manager._save_job(job)
                state = manager._load_state()
                state.detect_job_id = job.job_id
                state.detect_video_name = job.video_name
                state.detect_batch_index = item["batch_index"]
                manager._save_state(state)
                return job

            manager._launch_detection = fake_launch  # type: ignore[method-assign]

            manager.start_detection_batch(videos, model_path=model_path)

            self.assertEqual(launched, [("first.mp4", 1)])
            self.assertEqual(
                [(item["video_name"], item["batch_index"]) for item in manager._detect_queue],
                [("second.mp4", 2), ("third.mp4", 3)],
            )
            state = manager._load_state()
            self.assertEqual(state.detect_batch_total, 3)
            self.assertEqual(state.detect_batch_index, 1)
            self.assertEqual(state.detect_video_name, "first.mp4")

    def test_start_next_queued_video_updates_active_job(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = DetectPipelineManager(
                root=root / "pipeline",
                dataset_root=root / "dataset",
                config_dir=root / "config",
            )
            manager._batch_total = 2
            manager._batch_done = 1
            manager._detect_queue.append({"video_name": "second.mp4", "batch_index": 2})

            state = manager._load_state()
            state.detect_status = "running"
            state.detect_job_id = "first-job"
            state.detect_queue_pending = 1
            state.detect_batch_total = 2
            state.detect_batch_done = 1
            state.detect_batch_index = 1
            manager._save_state(state)

            launched: list[str] = []

            def fake_launch(item):
                launched.append(item["video_name"])
                job = DetectJob(
                    job_id="second-job",
                    video_name=item["video_name"],
                    created_at="2026-06-29T00:00:00+00:00",
                    status="running",
                )
                manager._save_job(job)
                next_state = manager._load_state()
                next_state.detect_status = "running"
                next_state.detect_job_id = job.job_id
                next_state.detect_video_name = job.video_name
                next_state.detect_queue_pending = len(manager._detect_queue)
                manager._save_state(next_state)
                return job

            manager._launch_detection = fake_launch  # type: ignore[method-assign]

            manager._start_next_queued_or_finish(success=True)

            self.assertEqual(launched, ["second.mp4"])
            state = manager._load_state()
            self.assertEqual(state.detect_status, "running")
            self.assertEqual(state.detect_job_id, "second-job")
            self.assertEqual(state.detect_video_name, "second.mp4")
            self.assertEqual(state.detect_batch_index, 2)
            self.assertEqual(state.detect_queue_pending, 0)
            self.assertEqual(state.detect_batch_done, 1)


if __name__ == "__main__":
    unittest.main()
