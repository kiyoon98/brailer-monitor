from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "output" / "sea_area_20260128_0300_0600.csv"
PNG_PATH = ROOT / "output" / "sea_area_20260128_0300_0600.png"


def main() -> None:
    rows: list[dict[str, object]] = []
    with CSV_PATH.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "time": datetime.strptime(row["time"], "%Y-%m-%d %H:%M"),
                    "avg": float(row["avg_sea_percent"]),
                    "min": float(row["min_sea_percent"]),
                    "max": float(row["max_sea_percent"]),
                }
            )

    times = [row["time"] for row in rows]
    averages = [row["avg"] for row in rows]
    minimums = [row["min"] for row in rows]
    maximums = [row["max"] for row in rows]
    rolling = [
        sum(averages[max(0, index - 2) : index + 1]) / len(averages[max(0, index - 2) : index + 1])
        for index in range(len(averages))
    ]

    hourly: dict[str, list[float]] = {}
    for row in rows:
        label = row["time"].strftime("%H시")
        hourly.setdefault(label, []).append(row["avg"])
    hour_labels = list(hourly)
    hour_averages = [sum(values) / len(values) for values in hourly.values()]

    font_path = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    if Path(font_path).exists():
        font_manager.fontManager.addfont(font_path)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    figure, (timeline_ax, hourly_ax) = plt.subplots(
        2,
        1,
        figsize=(15, 8),
        gridspec_kw={"height_ratios": [3.2, 1]},
        constrained_layout=True,
    )
    figure.patch.set_facecolor("#f7f8fa")

    timeline_ax.set_facecolor("#ffffff")
    timeline_ax.fill_between(times, minimums, maximums, color="#7dd3fc", alpha=0.28, label="영상 내 최소-최대")
    timeline_ax.plot(times, averages, color="#007f73", marker="o", markersize=4, linewidth=1.8, label="5분 영상 평균")
    timeline_ax.plot(times, rolling, color="#e26d2f", linewidth=2.2, label="15분 이동평균")
    timeline_ax.axvline(datetime(2026, 1, 28, 5, 20), color="#c2413b", linestyle="--", linewidth=1.2)
    timeline_ax.annotate(
        "05:20 이후 뚜렷한 감소",
        xy=(datetime(2026, 1, 28, 5, 20), 47.51),
        xytext=(datetime(2026, 1, 28, 4, 48), 34),
        arrowprops={"arrowstyle": "->", "color": "#c2413b"},
        color="#8f2925",
        fontsize=10,
    )
    timeline_ax.set_title("JJR-102283 stream04 시간대별 바다 영역", fontsize=17, fontweight="bold", loc="left")
    timeline_ax.set_ylabel("바다 영역 (%)")
    timeline_ax.set_ylim(30, 85)
    timeline_ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))
    timeline_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    timeline_ax.grid(axis="y", color="#d9dee5", linewidth=0.8)
    timeline_ax.spines[["top", "right"]].set_visible(False)
    timeline_ax.legend(loc="lower left", frameon=False, ncols=3)

    colors = ["#4e9f92", "#e3a44a", "#c65d4b"]
    bars = hourly_ax.bar(hour_labels, hour_averages, color=colors, width=0.62)
    hourly_ax.set_facecolor("#ffffff")
    hourly_ax.set_title("시간대별 평균", fontsize=12, fontweight="bold", loc="left")
    hourly_ax.set_ylabel("평균 (%)")
    hourly_ax.set_ylim(0, 75)
    hourly_ax.grid(axis="y", color="#e3e6ea", linewidth=0.8)
    hourly_ax.spines[["top", "right"]].set_visible(False)
    for bar, value in zip(bars, hour_averages, strict=True):
        hourly_ax.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.1f}%", ha="center", fontweight="bold")

    PNG_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(PNG_PATH, dpi=180, facecolor=figure.get_facecolor())
    print(PNG_PATH)
    print(", ".join(f"{label}={value:.2f}%" for label, value in zip(hour_labels, hour_averages, strict=True)))


if __name__ == "__main__":
    main()
