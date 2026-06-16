"""Discover and download vessel videos from the internal Lake media server."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "lake_video.json"
DEFAULT_MINUTE_SLOTS = tuple(range(0, 60, 5))
DEFAULT_SECOND_SUFFIXES = ("16",)


@dataclass(frozen=True)
class LakeVideoConfig:
    profile_id: str
    label: str
    base_url: str
    file_prefix: str
    year: int
    minute_slots: tuple[int, ...]
    second_suffixes: tuple[str, ...]

    @classmethod
    def from_dict(cls, profile_id: str, payload: dict[str, Any]) -> LakeVideoConfig:
        base_url = str(payload.get("base_url", "")).strip()
        if base_url and not base_url.endswith("/"):
            base_url += "/"
        minute_slots = tuple(int(value) for value in payload.get("minute_slots", DEFAULT_MINUTE_SLOTS))
        suffixes = payload.get("second_suffixes")
        if suffixes is None:
            legacy = str(payload.get("second_suffix", DEFAULT_SECOND_SUFFIXES[0])).zfill(2)[-2:]
            suffixes = [legacy]
        normalized_suffixes = tuple(str(value).zfill(2)[-2:] for value in suffixes)
        if not normalized_suffixes:
            normalized_suffixes = DEFAULT_SECOND_SUFFIXES
        return cls(
            profile_id=profile_id,
            label=str(payload.get("label", profile_id)),
            base_url=base_url,
            file_prefix=str(payload.get("file_prefix", "JJR-102283_stream04")),
            year=int(payload.get("year", 2026)),
            minute_slots=minute_slots,
            second_suffixes=normalized_suffixes,
        )


from .video_detect import DetectionCancelled


def _read_config_payload(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_lake_profile_summaries(path: Path | None = None) -> list[dict[str, Any]]:
    """Return selectable Lake video naming profiles from config."""
    config_path = path or DEFAULT_CONFIG_PATH
    payload = _read_config_payload(config_path)
    if "profiles" in payload:
        default_profile = str(payload.get("default_profile", "default"))
        profiles = payload["profiles"]
        return [
            {
                "id": profile_id,
                "label": profiles[profile_id].get("label", profile_id),
                "file_prefix": profiles[profile_id].get("file_prefix", ""),
                "year": profiles[profile_id].get("year"),
                "default": profile_id == default_profile,
            }
            for profile_id in profiles
        ]
    if not payload:
        fallback = LakeVideoConfig.from_dict("default", {})
        return [
            {
                "id": "default",
                "label": fallback.label,
                "file_prefix": fallback.file_prefix,
                "year": fallback.year,
                "default": True,
            }
        ]
    return [
        {
            "id": "default",
            "label": payload.get("label", "default"),
            "file_prefix": payload.get("file_prefix", ""),
            "year": payload.get("year"),
            "default": True,
        }
    ]


def load_lake_video_config(path: Path | None = None, *, profile: str | None = None) -> LakeVideoConfig:
    config_path = path or DEFAULT_CONFIG_PATH
    payload = _read_config_payload(config_path)
    if not payload:
        return LakeVideoConfig.from_dict("default", {})

    if "profiles" in payload:
        profiles = payload["profiles"]
        profile_id = profile or str(payload.get("default_profile", next(iter(profiles))))
        if profile_id not in profiles:
            known = ", ".join(sorted(profiles))
            raise ValueError(f"알 수 없는 Lake 프로필: {profile_id} (사용 가능: {known})")
        return LakeVideoConfig.from_dict(profile_id, profiles[profile_id])

    return LakeVideoConfig.from_dict("default", payload)


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


def build_filename(
    hour_dt: datetime,
    minute: int,
    config: LakeVideoConfig,
    *,
    second_suffix: str | None = None,
) -> str:
    yymmdd = hour_dt.strftime("%y%m%d")
    suffix = second_suffix or config.second_suffixes[0]
    hhmmss = f"{hour_dt.hour:02d}{minute:02d}{str(suffix).zfill(2)[-2:]}"
    return f"{config.file_prefix}_{yymmdd}_{hhmmss}.mp4"


def build_folder_path(hour_dt: datetime) -> str:
    return f"{hour_dt.month:02d}/{hour_dt.day:02d}/{hour_dt.hour:02d}/"


def _iter_slot_suffixes(config: LakeVideoConfig) -> Iterable[str]:
    return config.second_suffixes


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
            for suffix in _iter_slot_suffixes(config):
                filename = build_filename(hour_dt, minute, config, second_suffix=suffix)
                url = f"{config.base_url}{folder}{filename}"
                videos.append(
                    {
                        "filename": filename,
                        "url": url,
                        "folder": folder,
                        "hour_label": hour_dt.strftime("%m-%d %H:00"),
                        "profile": config.profile_id,
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
    config = config or load_lake_video_config()
    if not check_exists:
        return list_candidate_videos(
            start_month=start_month,
            start_day=start_day,
            start_hour=start_hour,
            end_month=end_month,
            end_day=end_day,
            end_hour=end_hour,
            config=config,
        )

    found: list[dict[str, str]] = []
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
            matched: dict[str, str] | None = None
            for suffix in _iter_slot_suffixes(config):
                filename = build_filename(hour_dt, minute, config, second_suffix=suffix)
                url = f"{config.base_url}{folder}{filename}"
                if probe_video_exists(url):
                    matched = {
                        "filename": filename,
                        "url": url,
                        "folder": folder,
                        "hour_label": hour_dt.strftime("%m-%d %H:00"),
                        "profile": config.profile_id,
                    }
                    break
            if matched is not None:
                found.append(matched)
            else:
                logger.debug(
                    "Lake video missing for %s %02d:%02d (suffixes=%s)",
                    hour_dt.strftime("%m-%d %H"),
                    hour_dt.hour,
                    minute,
                    ",".join(config.second_suffixes),
                )
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
