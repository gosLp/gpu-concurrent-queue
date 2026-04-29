#!/usr/bin/env python3
"""
Generate the paper RT 1x4 relative plot.

Input:
    ../rt_queue_perf.csv relative to this file:
        final/rt_queue_perf.csv

Output:
    final/results/figures/rt_queue_relative_vs_compaction_all_1x4.png

Figure:
    1 x 4 panels:
        MI210  Cornell
        MI210  Complex
        MI300A Cornell
        MI300A Complex

Metric:
    log10(MRays/s queue / MRays/s compaction)

Queues shown:
    GWFQ, GLFQ, WFQ, SFQ

No BQ-NL / BWD.
No absolute plots.
No 1x2 or 2x2 modes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "rt_queue_perf.csv"
OUT_DIR = ROOT / "results" / "figures"
OUT_PATH = OUT_DIR / "rt_queue_relative_vs_compaction_all_1x4.png"


# ============================================================
# Fixed config
# ============================================================

QUEUE_ORDER = ["GWFQ", "GLFQ", "WFQ", "SFQ"]
BASELINE_QUEUE = "COMPACT"
DEVICE_ORDER = ["MI210", "MI300A"]
SCENES = ["cornell", "complex"]
MAX_THREADS = 8192

QUEUE_LABELS = {
    "GWFQ": "G-WFQ",
    "GLFQ": "G-LFQ",
    "WFQ": "G-WFQ-YMC",
    "SFQ": "SFQ",
}

QUEUE_COLORS = {
    "GWFQ": "#d62728",
    "GLFQ": "#1f77b4",
    "WFQ": "#ff7f0e",
    "SFQ": "grey",
    "COMPACT": "green",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "legend.fontsize": 12,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "lines.linewidth": 1.6,
    "lines.markersize": 6,
    "figure.dpi": 600,
    "axes.linewidth": 1,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
})


# ============================================================
# Data loading
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def normalize_device(raw: object) -> str | None:
    val = str(raw).strip().upper()

    if "MI300A" in val or "MI300" in val or "GFX942" in val:
        return "MI300A"

    if "MI210" in val or "GFX90A" in val:
        return "MI210"

    return None


def normalize_queue(raw: object) -> str | None:
    val = str(raw).strip().upper().replace('"', "")

    if val.startswith("COMPACT") or val.startswith("COMPACTION"):
        return "COMPACT"

    if val.startswith("GWFQ"):
        return "GWFQ"

    if val.startswith("GLFQ"):
        return "GLFQ"

    # Keep this before generic checks if your CSV has exact WFQ.
    if val.startswith("WFQ"):
        return "WFQ"

    if val.startswith("SFQ"):
        return "SFQ"

    # Explicitly ignore BQ-NL/BWD and anything else.
    return None


def normalize_scene(raw: object) -> str | None:
    val = str(raw).strip().lower()

    if "cornell" in val or val == "1":
        return "cornell"

    if "complex" in val or "sphere" in val or val == "0":
        return "complex"

    return val if val in {"cornell", "complex"} else None


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise SystemExit(f"ERROR: missing RT CSV: {csv_path}")

    df = normalize_columns(pd.read_csv(csv_path))

    required = {
        "DEVICE",
        "QUEUE",
        "SCENE",
        "THREADS",
        "BOUNCES",
        "MRAYS_PER_S",
        "TOTAL_TIME_MS",
    }

    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"ERROR: {csv_path} missing columns: {missing}")

    df["DEVICE"] = df["DEVICE"].map(normalize_device)
    df["QUEUE"] = df["QUEUE"].map(normalize_queue)
    df["SCENE"] = df["SCENE"].map(normalize_scene)

    df["THREADS"] = pd.to_numeric(df["THREADS"], errors="coerce")
    df["BOUNCES"] = pd.to_numeric(df["BOUNCES"], errors="coerce")
    df["MRAYS_PER_S"] = pd.to_numeric(df["MRAYS_PER_S"], errors="coerce")
    df["TOTAL_TIME_MS"] = pd.to_numeric(df["TOTAL_TIME_MS"], errors="coerce")

    df = df[
        df["DEVICE"].isin(DEVICE_ORDER)
        & df["QUEUE"].isin(QUEUE_ORDER + [BASELINE_QUEUE])
        & df["SCENE"].isin(SCENES)
        & df["THREADS"].notna()
        & df["BOUNCES"].notna()
        & df["MRAYS_PER_S"].notna()
    ].copy()

    df = df[df["THREADS"] <= MAX_THREADS].copy()

    if df.empty:
        raise SystemExit("ERROR: no usable RT rows after filtering.")

    df["THREADS"] = df["THREADS"].astype(int)
    df["BOUNCES"] = df["BOUNCES"].astype(int)

    # Keep best throughput for repeated runs.
    idx = (
        df.groupby(
            ["DEVICE", "QUEUE", "SCENE", "BOUNCES", "THREADS"],
            dropna=False,
        )["MRAYS_PER_S"]
        .idxmax()
    )

    df = df.loc[idx].copy().reset_index(drop=True)

    return df


def choose_bounces(df: pd.DataFrame, scene: str) -> int:
    sdf = df[df["SCENE"] == scene]

    if sdf.empty:
        raise SystemExit(f"ERROR: no rows for scene={scene}")

    baseline = sdf[sdf["QUEUE"] == BASELINE_QUEUE]

    if not baseline.empty:
        return int(baseline["BOUNCES"].max())

    return int(sdf["BOUNCES"].max())


# ============================================================
# Plot helpers
# ============================================================

def to_pow2_label(threads: int) -> str:
    if threads <= 0:
        return str(threads)

    exp = int(np.log2(threads))

    if (1 << exp) == int(threads):
        return rf"$2^{{{exp}}}$"

    return str(threads)


def tick_label_for_log10(v: int) -> str:
    if v == 0:
        return "1"

    return rf"$10^{{{v}}}$"


def add_speed_direction_arrows(
    ax,
    y0: float,
    y_min: float,
    y_max: float,
    x_offset: float = -0.12,
) -> None:
    x = x_offset - 0.02

    up_span = 1.1 * (y_max - y0)
    down_span = 1.1 * (y0 - y_min)

    y_up = y0 + up_span
    y_down = y0 - down_span

    ax.annotate(
        "",
        xy=(x, y_up),
        xytext=(x, y0),
        xycoords=("axes fraction", "data"),
        textcoords=("axes fraction", "data"),
        arrowprops=dict(arrowstyle="-|>", lw=6.0, color="#2f6f3e"),
        annotation_clip=False,
        zorder=20,
    )

    ax.text(
        x + 0.11,
        y_up,
        "Faster",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="bottom",
        clip_on=False,
        fontsize=24,
        fontweight="bold",
        color="#2f6f3e",
    )

    ax.annotate(
        "",
        xy=(x, y_down),
        xytext=(x, y0),
        xycoords=("axes fraction", "data"),
        textcoords=("axes fraction", "data"),
        arrowprops=dict(arrowstyle="-|>", lw=6.0, color="#8b2e2e"),
        annotation_clip=False,
        zorder=20,
    )

    ax.text(
        x + 0.13,
        y_down,
        "Slower",
        transform=ax.get_yaxis_transform(),
        ha="right",
        va="top",
        clip_on=False,
        fontsize=24,
        fontweight="bold",
        color="#8b2e2e",
    )


def add_column_header(fig, ax, title: str) -> None:
    pos = ax.get_position()

    x0 = pos.x0
    width = pos.width
    y = pos.y1 + 0.010
    height = 0.060

    rect = Rectangle(
        (x0, y),
        width,
        height,
        transform=fig.transFigure,
        facecolor="#d8d8d8",
        edgecolor="#888888",
        linewidth=1.0,
        clip_on=False,
        zorder=20,
    )

    fig.add_artist(rect)

    fig.text(
        x0 + width / 2,
        y + height / 2,
        title,
        ha="center",
        va="center",
        fontsize=19,
        fontweight="bold",
        zorder=21,
    )


def add_left_axis_box(fig, ax, title: str) -> None:
    pos = ax.get_position()

    width = 0.038
    x = pos.x0 - 0.095
    y0 = pos.y0 - 0.05
    height = pos.height + 0.14

    rect = Rectangle(
        (x, y0),
        width,
        height,
        transform=fig.transFigure,
        facecolor="#ececec",
        edgecolor="#888888",
        linewidth=1.0,
        clip_on=False,
        zorder=20,
    )

    fig.add_artist(rect)

    fig.text(
        x + width / 2,
        y0 + height / 2,
        title,
        ha="center",
        va="center",
        rotation=90,
        fontsize=17,
        fontweight="bold",
        zorder=21,
    )


def dedup_legend(fig, axes) -> None:
    handles = []
    labels = []
    seen = set()

    for ax in axes:
        h, l = ax.get_legend_handles_labels()

        for handle, label in zip(h, l):
            if label in seen:
                continue

            handles.append(handle)
            labels.append(label)
            seen.add(label)

    if not handles:
        return

    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.55, 0.945),
        fontsize=18,
        frameon=True,
        borderpad=0.20,
    )

    leg.get_frame().set_facecolor("#efefef")
    leg.get_frame().set_edgecolor("#c0c0c0")
    leg.get_frame().set_linewidth(0.8)


# ============================================================
# Plot
# ============================================================

def plot_relative_1x4(df: pd.DataFrame) -> Path:
    bounces_by_scene = {
        scene: choose_bounces(df, scene)
        for scene in SCENES
    }

    panel_specs = [
        ("MI210", "cornell"),
        ("MI210", "complex"),
        ("MI300A", "cornell"),
        ("MI300A", "complex"),
    ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(23.5, 6.4),
        sharex=False,
        sharey=True,
    )

    fig.suptitle(
        "Raytracing Throughput Vs Stream Compaction",
        fontsize=28,
        fontweight="bold",
        y=0.995,
    )

    label_points = []
    width = 0.20

    for ci, (device, scene) in enumerate(panel_specs):
        bounces = bounces_by_scene[scene]
        ax = axes[ci]

        ax.set_facecolor("#f2f2f2" if scene == "cornell" else "#ffffff")

        sdf = df[
            (df["DEVICE"] == device)
            & (df["SCENE"] == scene)
            & (df["BOUNCES"] == bounces)
        ].copy()

        queue_rows = sdf[
            sdf["QUEUE"].isin(QUEUE_ORDER)
            & (sdf["THREADS"] <= MAX_THREADS)
        ]

        threads = sorted(queue_rows["THREADS"].unique().tolist())

        if not threads:
            ax.text(
                0.5,
                0.5,
                "No data",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")
            continue

        base_rows = sdf[sdf["QUEUE"] == BASELINE_QUEUE]

        if base_rows.empty:
            ax.text(
                0.5,
                0.5,
                "No COMPACT baseline",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")
            continue

        baseline_mrays = float(base_rows["MRAYS_PER_S"].max())

        if baseline_mrays <= 0:
            ax.text(
                0.5,
                0.5,
                "Invalid COMPACT baseline",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            ax.axis("off")
            continue

        x = np.arange(len(threads), dtype=float)

        for qi, queue in enumerate(QUEUE_ORDER):
            vals = []

            for thread_count in threads:
                row = sdf[
                    (sdf["QUEUE"] == queue)
                    & (sdf["THREADS"] == thread_count)
                ]

                if row.empty:
                    vals.append(np.nan)
                    continue

                mrays = float(row["MRAYS_PER_S"].iloc[0])

                if mrays <= 0:
                    vals.append(np.nan)
                    continue

                vals.append(np.log10(mrays / baseline_mrays))

            offset = (qi - 1.5) * width

            bars = ax.bar(
                x + offset,
                vals,
                width=width,
                color=QUEUE_COLORS[queue],
                label=QUEUE_LABELS[queue],
            )

            for bar_obj, rel in zip(bars, vals):
                if np.isfinite(rel):
                    label_points.append(
                        (
                            ax,
                            queue,
                            bar_obj.get_x() + bar_obj.get_width() / 2,
                            float(rel),
                        )
                    )

        ax.axhline(
            0.0,
            color=QUEUE_COLORS[BASELINE_QUEUE],
            linestyle="--",
            linewidth=2.0,
            label="Compaction = 1",
        )

        for i in range(len(x) - 1):
            ax.axvline(
                i + 0.5,
                color="#9a9a9a",
                linestyle=":",
                linewidth=0.9,
                alpha=0.45,
                zorder=1,
            )

        ax.set_xticks(x)
        ax.set_xticklabels([to_pow2_label(t) for t in threads], fontsize=16)

        ax.grid(True, axis="y", linestyle="--", alpha=0.35)
        ax.tick_params(axis="x", labelsize=18)
        ax.tick_params(axis="y", labelsize=18)

    y_min, y_max = -5, 3
    ticks = np.arange(y_min + 1, y_max, 1)
    labels = [tick_label_for_log10(int(t)) for t in ticks]

    for ci, ax in enumerate(axes):
        if not ax.has_data():
            continue

        ax.set_ylim(y_min, y_max)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=16)

        if ci == 0:
            add_speed_direction_arrows(
                ax,
                y0=0.0,
                y_min=y_min,
                y_max=y_max,
                x_offset=-0.12,
            )

    yr = max(1e-9, float(y_max - y_min))
    label_offset = 0.015 * yr

    for ax, queue, x_pos, rel in label_points:
        mult = 10 ** abs(rel)

        if queue == "SFQ" and mult >= 100.0:
            text = f"{int(mult):,}x"
        elif mult >= 10.0:
            text = f"{mult:.0f}x"
        else:
            text = f"{mult:.2f}x"

        if rel >= 0:
            y_text = min(rel + label_offset, y_max - 0.1)
            ax.text(
                x_pos,
                y_text,
                text,
                ha="center",
                va="bottom",
                fontsize=16.0,
                fontweight="bold",
                rotation=90,
            )
        else:
            y_text = rel - label_offset
            va = "top"

            if y_text < y_min + 0.2:
                y_text = y_min + 0.2
                va = "bottom"

            ax.text(
                x_pos,
                y_text,
                text,
                ha="center",
                va=va,
                fontsize=16.0,
                fontweight="bold",
                rotation=90,
            )

    fig.text(
        0.5,
        0.022,
        "Threads",
        ha="center",
        va="center",
        fontsize=28,
    )

    fig.subplots_adjust(
        left=0.09,
        right=0.992,
        bottom=0.1,
        top=0.78,
        wspace=0.03,
    )

    for ci, (device, scene) in enumerate(panel_specs):
        bounces = bounces_by_scene[scene]
        scene_name = "Cornell" if scene == "cornell" else "Complex"
        add_column_header(
            fig,
            axes[ci],
            f"{device}: {scene_name}, {bounces} reflections",
        )

    add_left_axis_box(
        fig,
        axes[0],
        "MRays/s vs Compaction Ratio (log10)",
    )

    dedup_legend(fig, axes)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=600, bbox_inches="tight")
    plt.close(fig)

    return OUT_PATH


def main() -> None:
    print(f"[RT-PLOT] Reading: {CSV_PATH}")
    df = load_data(CSV_PATH)

    out = plot_relative_1x4(df)
    print(f"[RT-PLOT] Saved: {out}")


if __name__ == "__main__":
    main()