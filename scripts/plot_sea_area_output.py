from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Patch


LINE_PATTERN = re.compile(
    r"absolute=(?P<absolute>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+) "
    r"sea=(?P<sea>[\d.]+)% state=(?P<state>\w+) "
    r"confidence=(?P<confidence>[\d.-]+) vessel=(?P<vessel>[\d.]+)%"
)


@dataclass(frozen=True)
class Sample:
    timestamp: datetime
    sea_percent: float
    vessel_percent: float
    confidence: float | None
    state: str


@dataclass(frozen=True)
class MinuteSummary:
    timestamp: datetime
    sample_count: int
    sea_avg: float
    sea_min: float
    sea_max: float
    vessel_avg: float
    confidence_avg: float | None
    state: str


def parse_samples(path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = LINE_PATTERN.search(line)
            if not match:
                continue
            confidence_text = match.group("confidence")
            samples.append(
                Sample(
                    timestamp=datetime.strptime(match.group("absolute"), "%Y-%m-%d %H:%M:%S.%f"),
                    sea_percent=float(match.group("sea")),
                    vessel_percent=float(match.group("vessel")),
                    confidence=float(confidence_text) if confidence_text != "--" else None,
                    state=match.group("state"),
                )
            )
    if not samples:
        raise ValueError(f"No sea-area samples found in {path}")
    return samples


def summarize_minutes(samples: list[Sample]) -> list[MinuteSummary]:
    grouped: dict[datetime, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.timestamp.replace(second=0, microsecond=0)].append(sample)

    summaries: list[MinuteSummary] = []
    for timestamp, minute_samples in sorted(grouped.items()):
        sea_values = [sample.sea_percent for sample in minute_samples]
        vessel_values = [sample.vessel_percent for sample in minute_samples]
        confidence_values = [sample.confidence for sample in minute_samples if sample.confidence is not None]
        state = Counter(sample.state for sample in minute_samples).most_common(1)[0][0]
        summaries.append(
            MinuteSummary(
                timestamp=timestamp,
                sample_count=len(minute_samples),
                sea_avg=sum(sea_values) / len(sea_values),
                sea_min=min(sea_values),
                sea_max=max(sea_values),
                vessel_avg=sum(vessel_values) / len(vessel_values),
                confidence_avg=(sum(confidence_values) / len(confidence_values) if confidence_values else None),
                state=state,
            )
        )
    return summaries


def state_spans(samples: list[Sample], state: str) -> list[tuple[datetime, datetime]]:
    spans: list[tuple[datetime, datetime]] = []
    start: datetime | None = None
    previous = samples[0].timestamp
    for sample in samples:
        if sample.state == state and start is None:
            start = sample.timestamp
        elif sample.state != state and start is not None:
            spans.append((start, previous + timedelta(seconds=1)))
            start = None
        previous = sample.timestamp
    if start is not None:
        spans.append((start, samples[-1].timestamp + timedelta(seconds=1)))
    return spans


def write_minute_csv(path: Path, summaries: list[MinuteSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time",
                "sample_count",
                "avg_sea_percent",
                "min_sea_percent",
                "max_sea_percent",
                "avg_vessel_percent",
                "avg_confidence",
                "dominant_state",
            ]
        )
        for row in summaries:
            writer.writerow(
                [
                    row.timestamp.isoformat(sep=" "),
                    row.sample_count,
                    f"{row.sea_avg:.4f}",
                    f"{row.sea_min:.4f}",
                    f"{row.sea_max:.4f}",
                    f"{row.vessel_avg:.4f}",
                    f"{row.confidence_avg:.4f}" if row.confidence_avg is not None else "",
                    row.state,
                ]
            )


def rolling_average(values: list[float], window: int) -> list[float]:
    return [
        sum(values[max(0, index - window + 1) : index + 1])
        / len(values[max(0, index - window + 1) : index + 1])
        for index in range(len(values))
    ]


def configure_korean_font() -> None:
    font_path = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def plot(samples: list[Sample], summaries: list[MinuteSummary], output_path: Path) -> None:
    configure_korean_font()
    times = [row.timestamp for row in summaries]
    sea_avg = [row.sea_avg for row in summaries]
    sea_min = [row.sea_min for row in summaries]
    sea_max = [row.sea_max for row in summaries]
    vessel_avg = [row.vessel_avg for row in summaries]
    confidence = [100.0 * (row.confidence_avg or 0.0) for row in summaries]
    sea_rolling = rolling_average(sea_avg, 5)
    encounter_spans = state_spans(samples, "encounter")

    hourly: dict[str, list[MinuteSummary]] = defaultdict(list)
    for row in summaries:
        hourly[row.timestamp.strftime("%H시")].append(row)
    hour_labels = list(hourly)
    hour_sea = [sum(item.sea_avg for item in rows) / len(rows) for rows in hourly.values()]
    hour_vessel = [sum(item.vessel_avg for item in rows) / len(rows) for rows in hourly.values()]

    figure, axes = plt.subplots(
        3,
        1,
        figsize=(16, 10),
        gridspec_kw={"height_ratios": [3.0, 1.7, 1.25]},
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#f5f7f9")
    sea_ax, signal_ax, hourly_ax = axes

    for axis in (sea_ax, signal_ax):
        for start, end in encounter_spans:
            axis.axvspan(start, end, color="#d95c4f", alpha=0.12, linewidth=0)

    sea_ax.set_facecolor("#ffffff")
    sea_ax.fill_between(times, sea_min, sea_max, color="#7dd3fc", alpha=0.3, label="1분 내 최소-최대")
    sea_ax.plot(times, sea_avg, color="#007f73", linewidth=1.3, label="1분 평균")
    sea_ax.plot(times, sea_rolling, color="#e46f2e", linewidth=2.4, label="5분 이동평균")
    sea_ax.set_title(
        "JJR-102283 stream04 시간대별 바다 영역 (Hybrid)",
        fontsize=17,
        fontweight="bold",
        loc="left",
    )
    sea_ax.set_ylabel("바다 영역 (%)")
    sea_ax.set_ylim(25, 70)
    sea_ax.grid(axis="y", color="#d9dee5", linewidth=0.8)
    sea_ax.spines[["top", "right"]].set_visible(False)
    handles, labels = sea_ax.get_legend_handles_labels()
    handles.append(Patch(facecolor="#d95c4f", alpha=0.18, label="encounter 상태"))
    labels.append("encounter 상태")
    sea_ax.legend(handles, labels, loc="lower left", frameon=False, ncols=4)

    signal_ax.set_facecolor("#ffffff")
    signal_ax.plot(times, vessel_avg, color="#c47a16", linewidth=1.7, label="선박 영역 1분 평균")
    signal_ax.plot(times, confidence, color="#596579", linewidth=1.3, linestyle="--", label="의미 분할 confidence")
    signal_ax.set_ylabel("비율 / 신뢰도 (%)")
    signal_ax.set_ylim(20, 100)
    signal_ax.grid(axis="y", color="#e0e4e9", linewidth=0.8)
    signal_ax.spines[["top", "right"]].set_visible(False)
    signal_ax.legend(loc="upper left", frameon=False, ncols=2)

    for axis in (sea_ax, signal_ax):
        axis.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

    hourly_ax.set_facecolor("#ffffff")
    positions = list(range(len(hour_labels)))
    width = 0.36
    sea_bars = hourly_ax.bar(
        [position - width / 2 for position in positions],
        hour_sea,
        width=width,
        color="#4e9f92",
        label="바다 영역",
    )
    vessel_bars = hourly_ax.bar(
        [position + width / 2 for position in positions],
        hour_vessel,
        width=width,
        color="#d6973d",
        label="선박 영역",
    )
    hourly_ax.set_title("시간대별 평균", fontsize=12, fontweight="bold", loc="left")
    hourly_ax.set_ylabel("평균 (%)")
    hourly_ax.set_xticks(positions, hour_labels)
    hourly_ax.set_ylim(0, 75)
    hourly_ax.grid(axis="y", color="#e3e6ea", linewidth=0.8)
    hourly_ax.spines[["top", "right"]].set_visible(False)
    hourly_ax.legend(loc="upper left", frameon=False, ncols=2)
    for bars in (sea_bars, vessel_bars):
        for bar in bars:
            value = bar.get_height()
            hourly_ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.8,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                fontweight="bold",
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor=figure.get_facecolor())


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot sea-area CLI text output")
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = (args.output or input_path.with_suffix(".png")).resolve()
    csv_path = (args.csv or input_path.with_suffix(".minute.csv")).resolve()
    samples = parse_samples(input_path)
    summaries = summarize_minutes(samples)
    write_minute_csv(csv_path, summaries)
    plot(samples, summaries, output_path)

    print(output_path)
    print(csv_path)
    print(f"samples={len(samples)} minutes={len(summaries)}")


if __name__ == "__main__":
    main()
