#!/usr/bin/env python3
"""
Generate the paper BFS centered relative-speed summary plot.

Input:
    final/bfs_bench_final.csv

Expected common schema:
    GPU, Graph, Queue, Threads, Time_med_s, TEPS_med_Gedges/s

Also accepts aliases:
    Time_ms, Time, Time (ms)

Output:
    final/results/figures/bfs_best_speedup_centered.png

Plot:
    One centered relative-speed summary figure.
    y = log2(Gunrock_time / Queue_time)

    y = 0  -> equal to Gunrock
    y > 0  -> faster than Gunrock
    y < 0  -> slower than Gunrock

Queues:
    WFQ, GLFQ, GWFQ, SFQ

GPUs:
    MI210, MI300A
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "bfs_bench_final.csv"
OUT_DIR = ROOT / "results" / "figures"
OUT_PATH = OUT_DIR / "bfs_best_speedup_centered.png"


# ============================================================
# Fixed config
# ============================================================

QUEUE_ORDER = ["WFQ", "GLFQ", "GWFQ", "SFQ"]
GPU_ORDER = ["MI210", "MI300A"]

QUEUE_STYLE = {
    "WFQ":  {"color": "#ff7f0e", "label": "G-WFQ-YMC"},
    "GLFQ": {"color": "#1f77b4", "label": "G-LFQ"},
    "GWFQ": {"color": "#d62728", "label": "G-WFQ"},
    "SFQ":  {"color": "grey",    "label": "SFQ"},
}

PREFERRED_GRAPHS = [
    "roadNet-CA",
    "belgium_osm",
    "hollywood-2009",
    "ak2010",
    "europe_osm",
    "delaunay_n21",
    "delaunay_n24",
    "road_usa",
    "kron_g500-logn21",
]

EXCLUDED_GRAPH_KEYWORDS = {"bips", "coauthors", "webbase"}

# Gunrock MI210 times in ms.
GUNROCK_MI210_MS = {
    "roadNet-CA": 113.252,
    "belgium_osm": 205.975,
    "hollywood-2009": 13.5045,
    "ak2010": 13.5045,
    "europe_osm": 2842.51,
    "delaunay_n21": 81.92,
    "delaunay_n24": 209.29,
    "road_usa": 821.136,
    "kron_g500-logn21": 37.9522,
}

# Gunrock MI300A times in ms.
GUNROCK_MI300A_MS = {
    "roadNet-CA": 68.6799,
    "belgium_osm": 135.742,
    "hollywood-2009": 13.06,
    "ak2010": 35.528,
    "europe_osm": 2006.91,
    "delaunay_n21": 98.4343,
    "delaunay_n24": 202.672,
    "road_usa": 616.37,
    "kron_g500-logn21": 51.5484,
}

GPU_TO_GUNROCK = {
    "MI210": GUNROCK_MI210_MS,
    "MI300A": GUNROCK_MI300A_MS,
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "figure.dpi": 600,
    "axes.linewidth": 1,
    "grid.linewidth": 0.5,
    "legend.frameon": True,
})


# ============================================================
# Data loading
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    rename = {}
    for c in out.columns:
        key = str(c).strip().upper().replace("_", " ")

        if key in {"GPU", "DEVICE"}:
            rename[c] = "GPU"
        elif key in {"GRAPH", "GRAPH NAME"}:
            rename[c] = "Graph"
        elif key in {"QUEUE", "ALGO", "ALGORITHM"}:
            rename[c] = "Queue"
        elif key in {"THREADS", "THREAD", "NUM THREADS", "NUMTHREADS"}:
            rename[c] = "Threads"
        elif key in {"TIME MED S", "TIME MED", "TIME MS", "TIME (MS)", "TIME"}:
            rename[c] = "Time_ms"
        elif key in {"TEPS MED GEDGES/S", "TEPS GEDGES PER S", "TEPS", "TEPS (GE/S)"}:
            rename[c] = "TEPS_GEdges_per_s"

    return out.rename(columns=rename)


def normalize_gpu(raw: object) -> str | None:
    s = str(raw).strip().upper()

    if "MI300A" in s or "MI300" in s or "GFX942" in s:
        return "MI300A"

    if "MI210" in s or "GFX90A" in s:
        return "MI210"

    return None


def normalize_graph(raw: object) -> str:
    s = str(raw).strip()

    if "/" in s or s.endswith(".mtx"):
        return Path(s).stem

    return s


def normalize_queue(raw: object) -> str | None:
    q = str(raw).strip().upper().replace('"', "")

    if q == "WFQ":
        return "WFQ"
    if q == "GLFQ":
        return "GLFQ"
    if q == "GWFQ":
        return "GWFQ"
    if q == "SFQ":
        return "SFQ"

    return None


def is_excluded_graph(graph: str) -> bool:
    g = graph.strip().lower()
    return any(k in g for k in EXCLUDED_GRAPH_KEYWORDS)


def load_bfs_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"ERROR: missing BFS CSV: {csv_path}")

    df = normalize_columns(pd.read_csv(csv_path))

    required = {"GPU", "Graph", "Queue", "Threads", "Time_ms"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise SystemExit(
            f"ERROR: {csv_path} missing columns after aliasing: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )

    df["GPU"] = df["GPU"].map(normalize_gpu)
    df["Graph"] = df["Graph"].map(normalize_graph)
    df["Queue"] = df["Queue"].map(normalize_queue)
    df["Threads"] = pd.to_numeric(df["Threads"], errors="coerce")
    df["Time_ms"] = pd.to_numeric(df["Time_ms"], errors="coerce")

    df = df[
        df["GPU"].isin(GPU_ORDER)
        & df["Queue"].isin(QUEUE_ORDER)
        & df["Graph"].notna()
        & df["Threads"].notna()
        & df["Time_ms"].notna()
        & (df["Time_ms"] > 0)
    ].copy()

    df["Threads"] = df["Threads"].astype(int)

    # Keep fastest repeated run per GPU/graph/queue/thread.
    df = df.sort_values(
        ["GPU", "Graph", "Queue", "Threads", "Time_ms"],
        ascending=[True, True, True, True, True],
    )

    df = df.drop_duplicates(
        subset=["GPU", "Graph", "Queue", "Threads"],
        keep="first",
    )

    if df.empty:
        raise SystemExit("ERROR: no usable BFS rows after filtering.")

    return df


def best_by_gpu_graph_queue(df: pd.DataFrame) -> pd.DataFrame:
    best = df.sort_values(
        ["GPU", "Graph", "Queue", "Time_ms"],
        ascending=[True, True, True, True],
    )

    best = best.groupby(
        ["GPU", "Graph", "Queue"],
        as_index=False,
    ).first()

    return best


def graphs_with_complete_data(df: pd.DataFrame) -> list[str]:
    required_queues = set(QUEUE_ORDER)
    graphs = sorted(df["Graph"].unique().tolist())

    out = []

    for graph in graphs:
        if is_excluded_graph(graph):
            continue

        ok = True

        for gpu in GPU_ORDER:
            d = df[(df["GPU"] == gpu) & (df["Graph"] == graph)]
            present = set(d["Queue"].dropna().unique().tolist())

            if not required_queues.issubset(present):
                ok = False
                break

            if graph not in GPU_TO_GUNROCK[gpu]:
                ok = False
                break

        if ok:
            out.append(graph)

    ordered = [g for g in PREFERRED_GRAPHS if g in out]
    ordered += [g for g in out if g not in ordered]

    if not ordered:
        raise SystemExit(
            "ERROR: no graph has complete data for both GPUs, all queues, and Gunrock baselines."
        )

    return ordered


# ============================================================
# Plot helpers
# ============================================================

def lighten_color(color: str, amount: float = 0.45) -> tuple[float, float, float]:
    c = np.array(mcolors.to_rgb(color))
    return tuple(np.clip(c + (1.0 - c) * amount, 0.0, 1.0))


def add_grey_title(ax, title: str) -> None:
    rect = Rectangle(
        (0, 1.01),
        1,
        0.105,
        transform=ax.transAxes,
        facecolor="#d8d8d8",
        edgecolor="#888888",
        linewidth=1.0,
        clip_on=False,
        zorder=10,
    )

    ax.add_patch(rect)

    ax.text(
        0.5,
        1.062,
        title,
        transform=ax.transAxes,
        fontsize=13,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=11,
    )


def add_speed_arrows(ax, y_min: float, y_max: float) -> None:
    x = 0.985
    y0 = 0.0

    up_color = "#2f6f3e"
    down_color = "#8b2e2e"

    y_up = y0 + 0.72 * (y_max - y0)
    y_down = y0 - 0.72 * (y0 - y_min)

    ax.annotate(
        "",
        xy=(x, y_up),
        xytext=(x, y0),
        xycoords=("axes fraction", "data"),
        textcoords=("axes fraction", "data"),
        arrowprops=dict(arrowstyle="-|>", lw=2.4, color=up_color),
        zorder=20,
    )

    ax.text(
        x + 0.012,
        y_up,
        "faster",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=11,
        fontweight="bold",
        color=up_color,
    )

    ax.annotate(
        "",
        xy=(x, y_down),
        xytext=(x, y0),
        xycoords=("axes fraction", "data"),
        textcoords=("axes fraction", "data"),
        arrowprops=dict(arrowstyle="-|>", lw=2.4, color=down_color),
        zorder=20,
    )

    ax.text(
        x + 0.018,
        y_down,
        "slower",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=down_color,
    )


def format_multiplier(mult: float) -> str:
    if not np.isfinite(mult) or mult <= 0:
        return ""

    if mult >= 100:
        return f"{int(round(mult))}x"

    if mult >= 10:
        return f"{mult:.0f}x"

    return f"{mult:.2f}x"


def format_tick(v: int) -> str:
    if v == 0:
        return "Gunrock"

    mag = 2 ** abs(v)

    if v > 0:
        return f"{int(mag)}x faster"

    return f"{int(mag)}x slower"


# ============================================================
# Plot
# ============================================================

def plot_centered_speedup(df: pd.DataFrame) -> Path:
    graphs = graphs_with_complete_data(df)
    best = best_by_gpu_graph_queue(df)

    fig, ax = plt.subplots(1, 1, figsize=(13.2, 5.25))

    x = np.arange(len(graphs))
    bar_width = 0.105
    queue_gap = 0.055

    queue_centers = np.arange(len(QUEUE_ORDER)) - (len(QUEUE_ORDER) - 1) / 2
    gpu_offsets = np.arange(len(GPU_ORDER)) - (len(GPU_ORDER) - 1) / 2
    queue_span = len(GPU_ORDER) * bar_width + queue_gap

    all_values: list[float] = []
    label_points: list[tuple[float, float, float]] = []

    for qi, queue in enumerate(QUEUE_ORDER):
        base_color = QUEUE_STYLE[queue]["color"]

        for gi, gpu in enumerate(GPU_ORDER):
            vals = []

            for graph in graphs:
                row = best[
                    (best["GPU"] == gpu)
                    & (best["Graph"] == graph)
                    & (best["Queue"] == queue)
                ]

                if row.empty:
                    vals.append(np.nan)
                    continue

                queue_time = float(row.iloc[0]["Time_ms"])
                gunrock_time = float(GPU_TO_GUNROCK[gpu].get(graph, np.nan))

                if not np.isfinite(queue_time) or not np.isfinite(gunrock_time):
                    vals.append(np.nan)
                    continue

                if queue_time <= 0 or gunrock_time <= 0:
                    vals.append(np.nan)
                    continue

                # Positive means faster than Gunrock.
                rel = float(np.log2(gunrock_time / queue_time))
                vals.append(rel)
                all_values.append(rel)

            xpos = (
                x
                + queue_centers[qi] * queue_span
                + gpu_offsets[gi] * bar_width
            )

            color = base_color if gpu == "MI210" else lighten_color(base_color, 0.45)
            hatch = None if gpu == "MI210" else "///"
            edgecolor = "black" if gpu == "MI300A" else None
            linewidth = 0.4 if gpu == "MI300A" else 0.0

            bars = ax.bar(
                xpos,
                vals,
                width=bar_width,
                color=color,
                edgecolor=edgecolor,
                linewidth=linewidth,
                hatch=hatch,
                label=f"{QUEUE_STYLE[queue]['label']} {gpu}",
            )

            for bar, rel in zip(bars, vals):
                if np.isfinite(rel):
                    label_points.append(
                        (
                            bar.get_x() + bar.get_width() / 2,
                            float(rel),
                            float(2 ** abs(rel)),
                        )
                    )

    ax.axhline(
        0.0,
        color="green",
        linestyle="-",
        linewidth=1.8,
        alpha=0.95,
    )

    if not all_values:
        raise SystemExit("ERROR: no finite relative speed values to plot.")

    ymin = min(min(all_values), -0.05)
    ymax = max(max(all_values), 0.05)
    span = max(0.25, ymax - ymin)
    pad = 0.20 * span

    ax.set_ylim(ymin - pad, ymax + pad)

    y_min, y_max = ax.get_ylim()
    add_speed_arrows(ax, y_min, y_max)

    tick_lo = int(np.floor(y_min))
    tick_hi = int(np.ceil(y_max))
    ticks = list(range(tick_lo, tick_hi + 1, 2))

    if 0 not in ticks:
        ticks = sorted(set(ticks + [0]))

    ax.set_yticks(ticks)
    ax.set_yticklabels([format_tick(t) for t in ticks])

    for tick_val, tick_label in zip(ticks, ax.get_yticklabels()):
        if tick_val == 0:
            tick_label.set_color("green")
            tick_label.set_fontweight("bold")

    y_range = max(1e-9, y_max - y_min)
    label_offset = 0.018 * y_range

    for xp, rel, mult in label_points:
        text = format_multiplier(mult)

        if rel >= 0:
            ax.text(
                xp,
                rel + label_offset,
                text,
                ha="center",
                va="bottom",
                fontsize=7.6,
                fontweight="bold",
                rotation=90,
                clip_on=False,
            )
        else:
            ax.text(
                xp,
                rel - label_offset,
                text,
                ha="center",
                va="top",
                fontsize=7.6,
                fontweight="bold",
                rotation=90,
                clip_on=False,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(graphs, rotation=22, ha="right")

    for sep in np.arange(len(graphs) - 1) + 0.5:
        ax.axvline(
            sep,
            color="#666666",
            linestyle=":",
            linewidth=0.9,
            alpha=0.55,
            zorder=0,
        )

    ax.grid(True, linestyle="--", alpha=0.38, axis="y")
    ax.set_xlabel("Graph")
    ax.set_ylabel("Relative speed vs Gunrock, log2 scale")

    add_grey_title(
        ax,
        "BFS Best Runtime Relative to Gunrock: Up = Faster, Down = Slower",
    )

    handles, labels = ax.get_legend_handles_labels()

    # Deduplicate legend.
    seen = set()
    handles2 = []
    labels2 = []

    for h, l in zip(handles, labels):
        if l in seen:
            continue
        handles2.append(h)
        labels2.append(l)
        seen.add(l)

    leg = ax.legend(
        handles2,
        labels2,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        ncol=2,
        fontsize=9.5,
        frameon=True,
    )

    leg.get_frame().set_facecolor("#f7f7f7")
    leg.get_frame().set_edgecolor("#c8c8c8")
    leg.get_frame().set_linewidth(0.8)

    fig.subplots_adjust(
        left=0.10,
        right=0.992,
        bottom=0.22,
        top=0.82,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return OUT_PATH


def main() -> None:
    print(f"[BFS-PLOT] Reading: {CSV_PATH}")
    df = load_bfs_data(CSV_PATH)

    out = plot_centered_speedup(df)
    print(f"[BFS-PLOT] Saved: {out}")


if __name__ == "__main__":
    main()