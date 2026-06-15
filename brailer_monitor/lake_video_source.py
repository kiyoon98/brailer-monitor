"""Discover and download vessel videos from the internal Lake media server."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "lake_video.json"


@dataclass(frozen=True)
class LakeVideoConfig:
    base_url: str
    file_prefix: str
    year: int
    minute_slots: tuple[int, ...]
    second_suffix: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LakeVideoConfig:
        base_url = str(payload.get("base_url", "")).strip()
        if base_url and not base_url.endswith("/"):
            base_url += "/"
        minute_slots = tuple(int(value) for value in payload.get("minute_slots", range(0, 60, 5)))
        return cls(
            base_url=base_url,
            file_prefix=str(payload.get("file_prefix", "JJR-102283_stream04")),
            year=int(payload.get("year", 2026)),
            minute_slots=minute_slots,
            second_suffix=str(payload.get("second_suffix", "16")).zfill(2)[-2:],
        )


from .video_detect import DetectionCancelled


def load_lake_video_config(path: Path | None = None) -> LakeVideoConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    if not config_path.exists():
        return LakeVideoConfig.from_dict({})
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return LakeVideoConfig.from_dict(payload)


def iter_hours_in_range(
    *,
    start_month: int,
    start_day: int,
    start_hour: int,
    end_month: int,
    end_day: int,
    end_hour: int,
    year: int,
) -> list[datetime]:
    start = datetime(year, start_month, start_day, start_hour)
    end = datetime(year, end_month, end_day, end_hour)
    if end < start:
        raise ValueError("종료 시각이 시작 시각보다 이릅니다.")

    hours: list[datetime] = []
    current = start
    while current <= end:
        hours.append(current)
        current += timedelta(hours=1)
    return hours


def build_filename(hour_dt: datetime, minute: int, config: LakeVideoConfig) -> str:
    yymmdd = hour_dt.strftime("%y%m%d")
    hhmmss = f"{hour_dt.hour:02d}{minute:02d}{config.second_suffix}"
    return f"{config.file_prefix}_{yymmdd}_{hhmmss}.mp4"


def build_folder_path(hour_dt: datetime) -> str:
    return f"{hour_dt.month:02d}/{hour_dt.day:02d}/{hour_dt.hour:02d}/"


def list_candidate_videos(
    *,
    start_month: int,
    start_day: int,
    start_hour: int,
    end_month: int,
    end_day: int,
    end_hour: int,
    config: LakeVideoConfig | None = None,
) -> list[dict[str, str]]:
    config = config or load_lake_video_config()
    videos: list[dict[str, str]] = []
    for hour_dt in iter_hours_in_range(
        start_month=start_month,
        start_day=start_day,
        start_hour=start_hour,
        end_month=end_month,
        end_day=end_day,
        end_hour=end_hour,
        year=config.year,
    ):
        folder = build_folder_path(hour_dt)
        for minute in config.minute_slots:
            filename = build_filename(hour_dt, minute, config)
            url = f"{config.base_url}{folder}{filename}"
            videos.append(
                {
                    "filename": filename,
                    "url": url,
                    "folder": folder,
                    "hour_label": hour_dt.strftime("%m-%d %H:00"),
                }
            )
    return videos


def probe_video_exists(url: str, *, timeout: float = 8.0) -> bool:
    for method, headers in (
        ("HEAD", {}),
        ("GET", {"Range": "bytes=0-0"}),
    ):
        request = urllib.request.Request(url, method=method, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if response.status in (200, 206):
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code in (200, 206):
                return True
        except Exception:
            continue
    return False


def discover_videos_in_range(
    *,
    start_month: int,
    start_day: int,
    start_hour: int,
    end_month: int,
    end_day: int,
    end_hour: int,
    config: LakeVideoConfig | None = None,
    check_exists: bool = True,
) -> list[dict[str, str]]:
    candidates = list_candidate_videos(
        start_month=start_month,
        start_day=start_day,
        start_hour=start_hour,
        end_month=end_month,
        end_day=end_day,
        end_hour=end_hour,
        config=config,
    )
    if not check_exists:
        return candidates

    found: list[dict[str, str]] = []
    for video in candidates:
        if probe_video_exists(video["url"]):
            found.append(video)
        else:
            logger.debug("Lake video missing: %s", video["url"])
    return found


def download_video(
    url: str,
    dest_path: Path,
    *,
    timeout: float = 120.0,
    should_cancel: Callable[[], bool] | None = None,
) -> None:
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        chunks: list[bytes] = []
        while True:
            if should_cancel and should_cancel():
                raise DetectionCancelled("Detection cancelled by user")
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        dest_path.write_bytes(b"".join(chunks))
