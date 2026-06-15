"""Parse recording start time from vessel video filenames."""

from __future__ import annotations

import re
from datetime import datetime, timedelta

# JJR-102283_stream04_260201_040016.mp4 -> 2026-02-01 04:00:00
_VIDEO_TIME_RE = re.compile(r"_(\d{6})_(\d{6})(?:\.|$)")


def parse_video_start_time(filename: str) -> datetime | None:
    """Return recording start time encoded in the filename, or None."""
    stem = filename.rsplit("/", 1)[-1]
    if stem.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        stem = stem.rsplit(".", 1)[0]

    match = _VIDEO_TIME_RE.search(stem)
    if not match:
        return None

    yymmdd, hhmmxx = match.groups()
    try:
        year = 2000 + int(yymmdd[:2])
        month = int(yymmdd[2:4])
        day = int(yymmdd[4:6])
        hour = int(hhmmxx[:2])
        minute = int(hhmmxx[2:4])
        return datetime(year, month, day, hour, minute, 0)
    except ValueError:
        return None


def absolute_frame_time(video_name: str, timestamp_sec: float) -> datetime | None:
    start = parse_video_start_time(video_name)
    if start is None:
        return None
    return start + timedelta(seconds=timestamp_sec)


def format_absolute_time(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    return dt.strftime("%Y-%m-%d %H:%M:%S")
