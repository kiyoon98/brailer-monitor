"""Discover and download vessel videos from the internal Lake media server."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "lake_video.json"
DEFAULT_MINUTE_OFFSETS = (0, 1, 2, 3, 4)
DEFAULT_MINUTE_SLOTS = tuple(range(0, 60, 5))
DEFAULT_SECOND_SUFFIXES = ("16",)
DEFAULT_PROBE_TIMEOUT_SEC = 2.0
DEFAULT_PROBE_WORKERS = 16

DEFAULT_BASE_HOST = "http://10.2.10.158:8041/media/em_data/"
DEFAULT_LAKE_COMPONENTS: dict[str, dict[str, Any]] = {
    "media": {
        "label": "미디어 폴더",
        "default": "lake_win",
        "options": ["lake_win", "lake_aurora", "lake_dream", "seibu", "pharostar"],
    },
    "year_folder": {
        "label": "연도 폴더",
        "default": "2026_decrypted",
        "options": ["2025_decrypted", "2026_decrypted", "2027_decrypted"],
    },
    "vessel": {
        "label": "선박",
        "default": "JJR-102283",
        "options": ["JJR-102283", "JJR-131066", "JJR-151069", "LAKE_AURORA", "JJR-211056", "seibu"],
    },
    "stream": {
        "label": "스트림",
        "default": "stream04",
        "options": ["stream01", "stream02", "stream03", "stream04"],
    },
    "suffix": {
        "label": "초 접미사",
        "default": "16",
        "options": ["16"],
    },
}


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


def load_lake_component_spec(path: Path | None = None) -> dict[str, Any]:
    """Return the selectable URL components (media/year/vessel/stream/suffix)."""
    payload = _read_config_payload(path or DEFAULT_CONFIG_PATH)
    base_host = str(payload.get("base_host", DEFAULT_BASE_HOST)).strip()
    if base_host and not base_host.endswith("/"):
        base_host += "/"
    if not base_host:
        base_host = DEFAULT_BASE_HOST
    raw_components = payload.get("components")
    components = raw_components if isinstance(raw_components, dict) and raw_components else DEFAULT_LAKE_COMPONENTS
    minute_offsets = [int(value) for value in payload.get("minute_offsets", DEFAULT_MINUTE_OFFSETS)]
    minute_slots = [int(value) for value in payload.get("minute_slots", DEFAULT_MINUTE_SLOTS)]
    return {
        "base_host": base_host,
        "components": components,
        "minute_offsets": minute_offsets,
        "minute_slots": minute_slots,
    }


def _selected_or_default(component: dict[str, Any], value: str | None, *, name: str) -> str:
    options = component.get("options", []) if isinstance(component, dict) else []
    default = component.get("default") if isinstance(component, dict) else None
    if not default and options:
        default = options[0]
    if value is None or value == "":
        return str(default or "")
    selected = str(value)
    allowed = {str(option) for option in options}
    if allowed and selected not in allowed:
        known = ", ".join(sorted(allowed))
        raise ValueError(f"알 수 없는 Lake {name}: {selected} (사용 가능: {known})")
    return selected


def _split_values(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _normalize_minute_slots(value: Any, default: Iterable[int]) -> tuple[int, ...]:
    raw_values = _split_values(value) or list(default)
    slots: list[int] = []
    for raw in raw_values:
        try:
            minute = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Lake 분 값이 올바르지 않습니다: {raw}") from exc
        if minute < 0 or minute > 59:
            raise ValueError(f"Lake 분 값은 0-59 사이여야 합니다: {minute}")
        if minute not in slots:
            slots.append(minute)
    if not slots:
        raise ValueError("Lake 분 후보가 비어 있습니다.")
    return tuple(slots)


def _normalize_minute_offsets(value: Any, default: Iterable[int]) -> tuple[int, ...]:
    raw_values = _split_values(value) or list(default)
    offsets: list[int] = []
    for raw in raw_values:
        try:
            offset = int(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Lake 시작 분 값이 올바르지 않습니다: {raw}") from exc
        if offset < 0 or offset > 4:
            raise ValueError(f"Lake 시작 분 값은 0-4 사이여야 합니다: {offset}")
        if offset not in offsets:
            offsets.append(offset)
    if not offsets:
        raise ValueError("Lake 시작 분 후보가 비어 있습니다.")
    return tuple(offsets)


def _minute_slots_from_offsets(offsets: Iterable[int]) -> tuple[int, ...]:
    slots: list[int] = []
    for offset in offsets:
        for minute in range(int(offset), 60, 5):
            if minute not in slots:
                slots.append(minute)
    return tuple(sorted(slots))


def _normalize_second_suffixes(value: Any, default: Iterable[str]) -> tuple[str, ...]:
    raw_values = _split_values(value) or list(default)
    suffixes: list[str] = []
    for raw in raw_values:
        text = str(raw).strip()
        if not text.isdigit():
            raise ValueError(f"Lake 초 suffix는 숫자여야 합니다: {text}")
        second = int(text)
        if second < 0 or second > 59:
            raise ValueError(f"Lake 초 suffix는 0-59 사이여야 합니다: {text}")
        suffix = f"{second:02d}"
        if suffix not in suffixes:
            suffixes.append(suffix)
    if not suffixes:
        raise ValueError("Lake 초 suffix가 비어 있습니다.")
    return tuple(suffixes)


def build_lake_config_from_selection(
    selection: dict[str, Any] | None,
    *,
    spec: dict[str, Any] | None = None,
    path: Path | None = None,
) -> LakeVideoConfig:
    """Compose a LakeVideoConfig from selected URL components."""
    spec = spec or load_lake_component_spec(path)
    components = spec["components"]
    sel = selection or {}

    media = _selected_or_default(components.get("media", {}), sel.get("media"), name="media")
    year_folder = _selected_or_default(
        components.get("year_folder", {}),
        sel.get("year_folder"),
        name="year_folder",
    )
    vessel = _selected_or_default(components.get("vessel", {}), sel.get("vessel"), name="vessel")
    stream = _selected_or_default(components.get("stream", {}), sel.get("stream"), name="stream")

    suffix_component = components.get("suffix", {}) if isinstance(components.get("suffix"), dict) else {}
    default_suffix = suffix_component.get("default")
    suffix_options = suffix_component.get("options")
    suffix_default = [default_suffix] if default_suffix not in (None, "") else suffix_options or DEFAULT_SECOND_SUFFIXES
    suffixes = _normalize_second_suffixes(sel.get("second_suffixes") or sel.get("second_suffix"), suffix_default)
    if sel.get("minute_slots") not in (None, ""):
        minute_slots = _normalize_minute_slots(sel.get("minute_slots"), spec["minute_slots"])
    else:
        offsets = _normalize_minute_offsets(
            sel.get("minute_offsets"),
            spec.get("minute_offsets", DEFAULT_MINUTE_OFFSETS),
        )
        minute_slots = _minute_slots_from_offsets(offsets)

    base_url = f"{spec['base_host']}{media}/{year_folder}/"
    digits = "".join(ch for ch in str(year_folder) if ch.isdigit())[:4]
    year = int(digits) if len(digits) == 4 else datetime.now().year
    label = f"{media}/{year_folder}/{vessel}_{stream}"

    return LakeVideoConfig(
        profile_id=label,
        label=label,
        base_url=base_url,
        file_prefix=f"{vessel}_{stream}",
        year=year,
        minute_slots=minute_slots,
        second_suffixes=suffixes,
    )


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


def probe_video_exists(url: str, *, timeout: float = DEFAULT_PROBE_TIMEOUT_SEC) -> bool:
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


def _candidate_minute_key(candidate: dict[str, str]) -> str:
    stem = Path(candidate["filename"]).stem
    return f"{candidate.get('folder', '')}:{stem[:-2]}"


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
    if not candidates:
        return []

    found_by_minute: dict[str, tuple[int, dict[str, str]]] = {}
    max_workers = min(DEFAULT_PROBE_WORKERS, len(candidates))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                probe_video_exists,
                candidate["url"],
                timeout=DEFAULT_PROBE_TIMEOUT_SEC,
            ): (index, candidate)
            for index, candidate in enumerate(candidates)
        }
        for future in as_completed(futures):
            index, candidate = futures[future]
            try:
                exists = future.result()
            except Exception:
                exists = False
            if not exists:
                continue
            key = _candidate_minute_key(candidate)
            current = found_by_minute.get(key)
            if current is None or index < current[0]:
                found_by_minute[key] = (index, candidate)

    return [candidate for _, candidate in sorted(found_by_minute.values(), key=lambda item: item[0])]


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
