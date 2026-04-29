#!/usr/bin/env python3
"""
Create a compact 2x4 throughput plot from benchmark_results.csv.

Layout:
    columns = [Balanced, 25% Prod, 50% Prod, 75% Prod]
    rows    = [MI210, MI300A]

Design goals:
- single shared legend
- single shared x-axis label
- single shared y-axis label
- one header box per column (only once at top)
- one row label box per GPU (only once at left)
- alternating subtle column shading:
    Balanced + 50% Prod   -> lightly shaded
    25% Prod + 75% Prod   -> white
"""

import argparse
import csv
import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# ----------------------------
# Config
# ----------------------------

QUEUE_ORDER = ["SFQ", "WFQ", "GWFQ", "GLFQ"]

QUEUE_COLORS = {
    "SFQ": "gray",
    # "BQ": "#948979",
    # "BWD": "#222831",
    "WFQ": "#f39c12",   # orange
    "GWFQ": "#d62728",  # red
    "GLFQ": "#1f77b4",  # blue
}

QUEUE_LABELS = {
    "SFQ": "SFQ",
    # "BQ": "BQ",
    # "BWD": "BWD",
    "WFQ": "G-WFQ-YMC",
    "GWFQ": "G-WFQ",
    "GLFQ": "G-LFQ",
}

QUEUE_MARKERS = {
    "SFQ": "^",
    # "BQ": "v",
    # "BWD": "+",
    "WFQ": "s",
    "GWFQ": "D",
    "GLFQ": "o",
}

MAX_THREADS = 32768
LINE_WIDTH = 2.6
MARKER_SIZE = 6.5
MARKER_EDGE_WIDTH = 1.0

TITLE_FONTSIZE = 20
LEGEND_FONTSIZE = 16
ROW_LABEL_FONTSIZE = 18
COL_LABEL_FONTSIZE = 11
AXIS_LABEL_FONTSIZE = 20
TICK_FONTSIZE = 13

# Column order for the new 2x4 design
PANEL_DEFS = [
    ("balanced", -1, "Balanced Kernel 1:1"),
    ("split", 25, "25% Producers / 75% Consumers"),
    ("split", 50, "50% Producers / 50% Consumers"),
    ("split", 75, "75% Producers / 25% Consumers"),
]

GPU_ROWS = ["MI210", "MI300A"]

# Column background style
SHADED_COLUMNS = {0, 2}   # Balanced, 50% Prod
# SHADE_FACE = "#f2f2f2"
SHADE_FACE = "#f9f9f9"
WHITE_FACE = "#ffffff"
HEADER_SHADE = "#d8d8d8"
HEADER_WHITE = "#fbfbfb"
HEADER_EDGE = "#444444"


# ----------------------------
# Helpers
# ----------------------------

def _norm_gpu(gpu_name: str) -> str:
    g = gpu_name.strip().upper()
    if g == "MI300A":
        return "MI300A"
    if g == "MI210":
        return "MI210"
    return g


def _norm_queue(queue_name: str) -> str:
    return queue_name.strip().strip('"').upper()


def _load_data(csv_path: str, metric: str):
    """
    Returns:
        data[gpu][(mode, producer_percent)][queue][thread] = average metric
    """
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float))))
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"gpu_name", "queue", "mode", "producer_percent", "threads", metric}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            gpu = _norm_gpu(row["gpu_name"])
            if gpu not in {"MI210", "MI300A"}:
                continue

            queue = _norm_queue(row["queue"])
            if queue not in QUEUE_ORDER:
                continue

            mode = row["mode"].strip().lower()
            if mode == "balanced":
                pp = -1
            elif mode == "split":
                try:
                    pp = int(float(row["producer_percent"]))
                except Exception:
                    continue
            else:
                continue

            try:
                threads = int(float(row["threads"]))
                val = float(row[metric])
            except (TypeError, ValueError):
                continue

            if threads > MAX_THREADS:
                continue

            sums[gpu][(mode, pp)][queue][threads] += val
            counts[gpu][(mode, pp)][queue][threads] += 1

    data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for gpu, gpu_data in sums.items():
        for mode_pp, mode_data in gpu_data.items():
            for queue, queue_data in mode_data.items():
                for thread, total in queue_data.items():
                    n = counts[gpu][mode_pp][queue][thread]
                    if n > 0:
                        data[gpu][mode_pp][queue][thread] = total / n

    return data


def _set_log2_x(ax, x_vals):
    x_vals = sorted(set(int(x) for x in x_vals if x is not None))
    if not x_vals:
        return
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_vals)
    ax.set_xticklabels(
        [rf"$2^{{{int(math.log2(x))}}}$" for x in x_vals],
        fontsize=TICK_FONTSIZE
    )


def _collect_column_shared_ylims(data):
    """
    One shared y-limit per column, shared across MI210 + MI300A.
    """
    ylims = []
    for mode, pp, _ in PANEL_DEFS:
        vals = []
        key = (mode, pp)

        for gpu in GPU_ROWS:
            panel = data.get(gpu, {}).get(key, {})
            for q in QUEUE_ORDER:
                for t, v in panel.get(q, {}).items():
                    if int(t) <= MAX_THREADS and v is not None and v > 0:
                        vals.append(v)

        if not vals:
            ylims.append((1e-3, 1.0))
            continue

        ymin = max(1e-3, min(vals) * 0.75)
        ymax = max(vals) * 1.35
        ylims.append((ymin, ymax))

    return ylims


def _get_metric_label(metric: str) -> str:
    return {
        "succ_mops": "Successful Ops (Mops/s)",
        "enq_mops": "Enqueue Throughput (Mops/s)",
        "deq_mops": "Dequeue Throughput (Mops/s)",
        "empty_mops": "Empty Rate (Mops/s)",
        "fail_mops": "Fail Rate (Mops/s)",
    }[metric]


def _add_column_headers(fig, axes):
    """
    Add one header box above each column.
    """
    for col, (_, _, title) in enumerate(PANEL_DEFS):
        top_ax = axes[0, col]
        pos = top_ax.get_position()

        x0 = pos.x0
        width = pos.width
        y = pos.y1 + 0.012
        height = 0.045

        face = HEADER_SHADE if col in SHADED_COLUMNS else HEADER_WHITE

        rect = Rectangle(
            (x0, y),
            width,
            height,
            transform=fig.transFigure,
            facecolor=face,
            edgecolor=HEADER_EDGE,
            linewidth=1.6,
            clip_on=False,
            zorder=10,
        )
        fig.add_artist(rect)

        fig.text(
            x0 + width / 2,
            y + height / 2,
            title,
            ha="center",
            va="center",
            fontsize=COL_LABEL_FONTSIZE,
            fontweight="bold",
            zorder=21,
        )


def _add_row_labels(fig, axes):
    """
    Add one vertical label box per GPU row on the left.
    """
    for row, gpu in enumerate(GPU_ROWS):
        left_ax = axes[row, 0]
        pos = left_ax.get_position()

        width = 0.030
        x = pos.x0 - 0.062
        y0 = pos.y0
        height = pos.height

        rect = Rectangle(
            (x, y0),
            width,
            height,
            transform=fig.transFigure,
            facecolor="#f0f0f0",
            edgecolor=HEADER_EDGE,
            linewidth=1.6,
            clip_on=False,
            zorder=20,
        )
        fig.add_artist(rect)

        fig.text(
            x + width / 2,
            y0 + height / 2,
            gpu,
            ha="center",
            va="center",
            rotation=90,
            fontsize=ROW_LABEL_FONTSIZE,
            fontweight="bold",
            zorder=21,
        )


# ----------------------------
# Plot
# ----------------------------

def plot_compact_2x4(data, metric, out_path, title):
    metric_label = _get_metric_label(metric)
    ylims = _collect_column_shared_ylims(data)

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4))
    plt.subplots_adjust(
        left=0.115,
        right=0.992,
        bottom=0.115,
        top=0.82,
        wspace=0.045,
        hspace=0.07,
    )

    legend_handles = []
    legend_labels = []

    for row, gpu in enumerate(GPU_ROWS):
        for col, (mode, pp, _) in enumerate(PANEL_DEFS):
            ax = axes[row, col]
            panel_key = (mode, pp)
            panel_data = data.get(gpu, {}).get(panel_key, {})

            # alternating subtle background treatment
            ax.set_facecolor(SHADE_FACE if col in SHADED_COLUMNS else WHITE_FACE)

            all_threads = sorted({
                int(t)
                for q in QUEUE_ORDER
                for t in panel_data.get(q, {}).keys()
            })

            for q in QUEUE_ORDER:
                tmap = panel_data.get(q, {})
                if not all_threads:
                    continue

                y = [float(tmap[t]) if t in tmap else np.nan for t in all_threads]

                line, = ax.plot(
                    all_threads,
                    y,
                    linestyle="-",
                    marker=QUEUE_MARKERS[q],
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    color=QUEUE_COLORS[q],
                    label=QUEUE_LABELS[q],
                    zorder=5,
                )

                if QUEUE_LABELS[q] not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(QUEUE_LABELS[q])

            _set_log2_x(ax, all_threads)
            ax.set_yscale("log")
            ax.set_ylim(ylims[col])

            # grid
            ax.grid(True, which="major", linestyle="--", alpha=0.30)
            ax.grid(True, which="minor", linestyle=":", alpha=0.16)

            # ticks
            ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

            # clean repeated labels
            if col != 0:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

            if row != 1:
                ax.tick_params(axis="x", which="both", labelbottom=False)

            # slightly stronger frame
            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("#555555")

    # global title
    fig.suptitle(title, fontsize=TITLE_FONTSIZE, fontweight="bold", y=0.98)

    # shared legend
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=True,
        framealpha=0.95,
        facecolor="#f9f9f9",
        edgecolor="#cfcfcf",
        fancybox=True,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=6,
        fontsize=LEGEND_FONTSIZE,
        handlelength=2.2,
        columnspacing=1.5,
    )

    # shared axis labels
    fig.text(0.5, 0.055, "Threads", ha="center", va="center", fontsize=AXIS_LABEL_FONTSIZE)
    fig.text(0.04, 0.49, metric_label, ha="center", va="center",
             rotation=90, fontsize=AXIS_LABEL_FONTSIZE)

    # headers + row labels
    _add_column_headers(fig, axes)
    _add_row_labels(fig, axes)

    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Create compact 2x4 queue throughput figure"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="../benchmark_results.csv",
        help="Path to benchmark_results.csv",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="succ_mops",
        choices=["succ_mops", "enq_mops", "deq_mops", "empty_mops", "fail_mops"],
        help="CSV metric column to plot",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="throughput_compact_2x4.png",
        help="Output image path",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Fixed-Duration Successful-Operation Throughput",
        help="Figure title",
    )
    args = parser.parse_args()

    data = _load_data(args.csv, args.metric)
    plot_compact_2x4(
        data=data,
        metric=args.metric,
        out_path=args.out,
        title=args.title,
    )
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()