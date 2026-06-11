"""Tracking helpers built on Ultralytics ByteTrack / BoT-SORT."""

from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection


@dataclass
class TrackState:
    track_id: int
    first_seen_frame: int
    last_seen_frame: int
    max_confidence: float = 0.0
    unload_crossed: bool = False
    entered_zone: bool = False
    completed: bool = False
    best_detection: Detection | None = None


@dataclass
class TrackManager:
    """Maintain per-track lifecycle for brailer transfer events."""

    active: dict[int, TrackState] = field(default_factory=dict)
    completed: list[TrackState] = field(default_factory=list)

    def update(self, frame_index: int, detections: list[Detection], unload_crossed_ids: set[int]) -> None:
        seen_ids: set[int] = set()
        for det in detections:
            if det.track_id is None:
                continue
            tid = det.track_id
            seen_ids.add(tid)
            state = self.active.get(tid)
            if state is None:
                state = TrackState(track_id=tid, first_seen_frame=frame_index, last_seen_frame=frame_index)
                self.active[tid] = state
            state.last_seen_frame = frame_index
            state.max_confidence = max(state.max_confidence, det.confidence)
            state.entered_zone = True
            if det.confidence >= (state.best_detection.confidence if state.best_detection else 0.0):
                state.best_detection = det
            if tid in unload_crossed_ids:
                state.unload_crossed = True

        finished: list[int] = []
        for tid, state in self.active.items():
            if tid in unload_crossed_ids and state.entered_zone and not state.completed:
                state.completed = True
                self.completed.append(state)
                finished.append(tid)
            elif tid not in seen_ids and frame_index - state.last_seen_frame > 30:
                finished.append(tid)

        for tid in finished:
            self.active.pop(tid, None)
