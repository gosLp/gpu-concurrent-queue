#!/usr/bin/env python3
"""
Create the paper 2x4 profiling plots.

Input layout, from repo root:

  benchmark_results.csv

  results/profiling/mi210/tb/gwfq/fifo0/mi210_gwfq_fifo0_stalls.csv
  results/profiling/mi210/tb/gwfq/fifo0/mi210_gwfq_fifo0_occ.csv

  results/profiling/mi300a/tb/gwfq/fifo0/mi300a_gwfq_fifo0_stalls.csv
  results/profiling/mi300a/tb/gwfq/fifo0/mi300a_gwfq_fifo0_occa.csv

Output:

  results/figures/mi210_profiling_compact_2x4.png
  results/figures/mi300a_profiling_compact_2x4.png

Figure layout:

  columns = [Balanced, 25% Prod, 50% Prod, 75% Prod]
  rows    = [WAIT/op, VALU/op]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle


# ============================================================
# Fixed artifact paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CSV = ROOT / "benchmark_results.csv"
PROFILE_ROOT = ROOT / "results" / "profiling"
FIGURE_DIR = ROOT / "results" / "figures"
FIFO_MODE = 0


# ============================================================
# Fixed 2x4 figure config
# ============================================================

QUEUES = ["SFQ", "GLFQ", "GWFQ", "WFQ"]

QUEUE_COLORS = {
    "SFQ": "#7f7f7f",
    "GLFQ": "#1f77b4",
    "GWFQ": "#d62728",
    "WFQ": "#ff7f0e",
}

QUEUE_MARKERS = {
    "SFQ": "o",
    "GLFQ": "s",
    "GWFQ": "D",
    "WFQ": "^",
}

QUEUE_LABELS = {
    "SFQ": "SFQ",
    "WFQ": "G-WFQ-YMC",
    "GWFQ": "G-WFQ",
    "GLFQ": "G-LFQ",
}

PANEL_DEFS = [
    ("balanced", "Balanced Kernel 1:1"),
    ("split_25", "25% Producers / 75% Consumers"),
    ("split_50", "50% Producers / 50% Consumers"),
    ("split_75", "75% Producers / 25% Consumers"),
]

SHADED_COLUMNS = {0, 2}
SHADE_FACE = "#f9f9f9"
WHITE_FACE = "#ffffff"
HEADER_SHADE = "#d8d8d8"
HEADER_WHITE = "#fbfbfb"
HEADER_EDGE = "#444444"

LEGEND_FONTSIZE = 16
COL_LABEL_FONTSIZE = 11
ROW_LABEL_FONTSIZE = 16
GPU_LABEL_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 13

LINE_WIDTH = 2.8
MARKER_SIZE = 6.5
MARKER_EDGE_WIDTH = 1.0

MAX_THREADS = 32768

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "legend.fontsize": 12,
    "xtick.labelsize": TICK_FONTSIZE,
    "ytick.labelsize": TICK_FONTSIZE,
    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,
    "figure.dpi": 600,
    "axes.linewidth": 1,
    "grid.linewidth": 0.5,
})


# ============================================================
# Normalization helpers
# ============================================================

def clean_columns(df: pd.DataFrame) -> None:
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[\u200b\ufeff]", "", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    clean_columns(df)
    return df


def norm_gpu_name(value: str) -> str:
    s = str(value).strip().upper()
    if "MI300" in s or "GFX942" in s:
        return "MI300A"
    if "MI210" in s or "GFX90A" in s:
        return "MI210"
    return s


def gpu_slug(gpu: str) -> str:
    g = norm_gpu_name(gpu)
    if g == "MI300A":
        return "mi300a"
    if g == "MI210":
        return "mi210"
    return g.lower()


def norm_queue_name(value: str) -> str:
    q = str(value).strip().strip('"').upper()
    q = q.replace("-", "").replace("_", "")

    if q in {"SFQ"}:
        return "SFQ"
    if q in {"WFQ", "GWFQYMC", "GYMC", "GWFQY"}:
        return "WFQ"
    if q in {"GWFQ"}:
        return "GWFQ"
    if q in {"GLFQ"}:
        return "GLFQ"

    return q


def queue_slug(queue: str) -> str:
    return norm_queue_name(queue).lower()


def benchmark_mode_columns(df: pd.DataFrame) -> tuple[str, str]:
    mode_col = "mode" if "mode" in df.columns else "test"
    pp_col = "producer_percent" if "producer_percent" in df.columns else "producer_ratio"

    if mode_col not in df.columns:
        raise ValueError("benchmark_results.csv needs either 'mode' or 'test' column")
    if pp_col not in df.columns:
        raise ValueError("benchmark_results.csv needs either 'producer_percent' or 'producer_ratio' column")

    return mode_col, pp_col


def profile_metric_path(gpu: str, queue: str, metric_tag: str) -> Path:
    g = gpu_slug(gpu)
    q = queue_slug(queue)

    return (
        PROFILE_ROOT
        / g
        / "tb"
        / q
        / f"fifo{FIFO_MODE}"
        / f"{g}_{q}_fifo{FIFO_MODE}_{metric_tag}.csv"
    )


def first_existing(paths: list[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# ============================================================
# Benchmark denominator: successful operations
# ============================================================

def load_success_ops(gpu: str, queue: str, level: str) -> pd.DataFrame:
    """
    Returns a dataframe with:

      Grid_Size, Success_ops

    Success_ops is computed from benchmark_results.csv:

      succ_mops * run_ms * 1000

    because:
      succ_mops = million successful ops / second
      run_ms    = milliseconds
    """

    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {BENCHMARK_CSV}")

    df = load_csv(BENCHMARK_CSV)
    if df.empty:
        return pd.DataFrame(columns=["Grid_Size", "Success_ops"])

    required = {"gpu_name", "queue", "threads", "succ_mops"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"benchmark_results.csv missing required columns: {sorted(missing)}")

    mode_col, pp_col = benchmark_mode_columns(df)

    df["_gpu_norm"] = df["gpu_name"].apply(norm_gpu_name)
    df["_queue_norm"] = df["queue"].apply(norm_queue_name)
    df["_mode_norm"] = df[mode_col].astype(str).str.strip().str.lower()

    gpu_norm = norm_gpu_name(gpu)
    queue_norm = norm_queue_name(queue)

    keep = (
        (df["_gpu_norm"] == gpu_norm)
        & (df["_queue_norm"] == queue_norm)
    )

    if level == "balanced":
        keep &= df["_mode_norm"].eq("balanced")
    else:
        target_pp = int(level.split("_", 1)[1])
        keep &= df["_mode_norm"].eq("split")
        keep &= pd.to_numeric(df[pp_col], errors="coerce").eq(target_pp)

    f = df.loc[keep].copy()
    if f.empty:
        print(f"[WARN] no benchmark rows for gpu={gpu_norm} queue={queue_norm} level={level}")
        return pd.DataFrame(columns=["Grid_Size", "Success_ops"])

    f["Grid_Size"] = pd.to_numeric(f["threads"], errors="coerce")
    f["succ_mops"] = pd.to_numeric(f["succ_mops"], errors="coerce")

    if "run_ms" in f.columns:
        f["run_ms"] = pd.to_numeric(f["run_ms"], errors="coerce")
    elif "elapsed_ms" in f.columns:
        f["run_ms"] = pd.to_numeric(f["elapsed_ms"], errors="coerce")
    else:
        raise ValueError("benchmark_results.csv needs 'run_ms' or 'elapsed_ms' for profile normalization")

    f["Success_ops"] = f["succ_mops"] * f["run_ms"] * 1000.0

    f = f[
        np.isfinite(f["Grid_Size"])
        & np.isfinite(f["Success_ops"])
        & (f["Grid_Size"] <= MAX_THREADS)
    ].copy()

    if f.empty:
        return pd.DataFrame(columns=["Grid_Size", "Success_ops"])

    f["Grid_Size"] = f["Grid_Size"].astype(int)

    out = (
        f.groupby("Grid_Size", as_index=False)["Success_ops"]
        .mean()
        .sort_values("Grid_Size")
    )

    return out


# ============================================================
# Rocprof numerator parsing
# ============================================================

def is_balanced_kernel(kernel_name: str) -> bool:
    k = str(kernel_name).lower()
    return "balanced_chunk_kernel" in k or "balanced_kernel" in k


def is_split_kernel(kernel_name: str) -> bool:
    k = str(kernel_name).lower()
    return "split_chunk_kernel" in k or "contention_kernel" in k


def filter_kernel_rows(df: pd.DataFrame, level: str) -> pd.DataFrame:
    """
    The throughput binary runs:

      balanced sweep
      split 25% sweep
      split 50% sweep
      split 75% sweep

    For split kernels, rocprof rows are separated into 25/50/75 by detecting
    when Grid_Size wraps from a larger value back to a smaller value.
    """

    if "Kernel_Name" not in df.columns:
        return df.iloc[0:0].copy()

    if level == "balanced":
        return df[df["Kernel_Name"].apply(is_balanced_kernel)].copy()

    split_df = df[df["Kernel_Name"].apply(is_split_kernel)].copy()
    if split_df.empty:
        return split_df

    if "Dispatch_ID" in split_df.columns:
        split_df["_dispatch_sort"] = pd.to_numeric(split_df["Dispatch_ID"], errors="coerce")
    else:
        split_df["_dispatch_sort"] = np.arange(len(split_df), dtype=float)

    split_df = split_df.sort_values("_dispatch_sort", kind="stable").copy()

    grids = pd.to_numeric(split_df["Grid_Size"], errors="coerce").to_numpy(dtype=float)

    sweep_idx = np.zeros(len(split_df), dtype=int)
    cur = 0

    for i in range(1, len(grids)):
        prev_g = grids[i - 1]
        cur_g = grids[i]

        if np.isfinite(prev_g) and np.isfinite(cur_g) and cur_g < prev_g:
            cur += 1

        sweep_idx[i] = cur

    sweep_to_percent = {0: 25, 1: 50, 2: 75}
    split_df["_producer_percent"] = [
        sweep_to_percent.get(int(s), np.nan) for s in sweep_idx
    ]

    target_pp = int(level.split("_", 1)[1])
    split_df = split_df[split_df["_producer_percent"].eq(target_pp)].copy()

    return split_df


def extract_valu(df_occ: pd.DataFrame) -> np.ndarray:
    if "SQ_INSTS_VALU" in df_occ.columns:
        return pd.to_numeric(df_occ["SQ_INSTS_VALU"], errors="coerce").to_numpy(dtype=float)

    v = np.zeros(len(df_occ), dtype=float)
    found = False

    for col in ["SQ_INSTS_VALU_INT32", "SQ_INSTS_VALU_INT64"]:
        if col in df_occ.columns:
            v += pd.to_numeric(df_occ[col], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            found = True

    if found:
        return v

    return np.full(len(df_occ), np.nan, dtype=float)


def load_profile_per_op(gpu: str, queue: str, level: str) -> pd.DataFrame:
    """
    Returns:

      Grid_Size, Wait_per_op, VALU_per_op

    WAIT/op = SQ_WAIT_ANY / successful_ops
    VALU/op = SQ_INSTS_VALU / successful_ops
    """

    stalls_path = profile_metric_path(gpu, queue, "stalls")
    occ_path = first_existing([
        profile_metric_path(gpu, queue, "occ"),
        profile_metric_path(gpu, queue, "occa"),
    ])

    if not stalls_path.exists():
        print(f"[WARN] missing stalls file: {stalls_path}")
        return pd.DataFrame(columns=["Grid_Size", "Wait_per_op", "VALU_per_op"])

    df_stalls_raw = load_csv(stalls_path)
    df_stalls = filter_kernel_rows(df_stalls_raw, level)

    if df_stalls.empty:
        print(f"[WARN] no stalls rows for gpu={gpu} queue={queue} level={level}")
        return pd.DataFrame(columns=["Grid_Size", "Wait_per_op", "VALU_per_op"])

    if "Grid_Size" not in df_stalls.columns or "SQ_WAIT_ANY" not in df_stalls.columns:
        print(f"[WARN] stalls file lacks Grid_Size or SQ_WAIT_ANY: {stalls_path}")
        return pd.DataFrame(columns=["Grid_Size", "Wait_per_op", "VALU_per_op"])

    df_stalls["Grid_Size"] = pd.to_numeric(df_stalls["Grid_Size"], errors="coerce")
    df_stalls["SQ_WAIT_ANY"] = pd.to_numeric(df_stalls["SQ_WAIT_ANY"], errors="coerce")

    stalls_g = (
        df_stalls[["Grid_Size", "SQ_WAIT_ANY"]]
        .dropna()
        .groupby("Grid_Size", as_index=False)
        .mean(numeric_only=True)
    )

    if occ_path is not None and occ_path.exists():
        df_occ_raw = load_csv(occ_path)
        df_occ = filter_kernel_rows(df_occ_raw, level)

        if not df_occ.empty and "Grid_Size" in df_occ.columns:
            df_occ = df_occ.copy()
            df_occ["Grid_Size"] = pd.to_numeric(df_occ["Grid_Size"], errors="coerce")
            df_occ["VALU"] = extract_valu(df_occ)

            occ_g = (
                df_occ[["Grid_Size", "VALU"]]
                .dropna(subset=["Grid_Size"])
                .groupby("Grid_Size", as_index=False)
                .mean(numeric_only=True)
            )
        else:
            occ_g = pd.DataFrame({
                "Grid_Size": stalls_g["Grid_Size"].to_numpy(),
                "VALU": np.nan,
            })
    else:
        print(f"[WARN] missing occ/occa file for gpu={gpu} queue={queue}")
        occ_g = pd.DataFrame({
            "Grid_Size": stalls_g["Grid_Size"].to_numpy(),
            "VALU": np.nan,
        })

    ops_g = load_success_ops(gpu, queue, level)

    if ops_g.empty:
        return pd.DataFrame(columns=["Grid_Size", "Wait_per_op", "VALU_per_op"])

    merged = (
        stalls_g
        .merge(occ_g, on="Grid_Size", how="left")
        .merge(ops_g, on="Grid_Size", how="inner")
    )

    if merged.empty:
        return pd.DataFrame(columns=["Grid_Size", "Wait_per_op", "VALU_per_op"])

    denom = merged["Success_ops"].replace(0, np.nan)

    merged["Wait_per_op"] = merged["SQ_WAIT_ANY"] / denom
    merged["VALU_per_op"] = merged["VALU"] / denom

    out = merged[["Grid_Size", "Wait_per_op", "VALU_per_op"]].copy()
    out = out[np.isfinite(out["Grid_Size"])]
    out["Grid_Size"] = out["Grid_Size"].astype(int)
    out = out[out["Grid_Size"] <= MAX_THREADS]
    out = out.sort_values("Grid_Size")

    return out


def build_level_data(gpu: str, level: str) -> dict[str, pd.DataFrame]:
    return {
        queue: load_profile_per_op(gpu, queue, level)
        for queue in QUEUES
    }


# ============================================================
# Plot helpers
# ============================================================

def set_log2_x(ax, x_vals) -> None:
    vals = sorted(set(int(x) for x in x_vals if pd.notna(x)))

    if not vals:
        return

    ax.set_xscale("log", base=2)
    ax.set_xticks(vals)
    ax.set_xticklabels(
        [rf"$2^{{{int(round(np.log2(x)))}}}$" for x in vals],
        fontsize=TICK_FONTSIZE,
    )


def collect_positive_values(level_data: dict, metric: str) -> np.ndarray:
    vals = []

    for panel_data in level_data.values():
        for df in panel_data.values():
            if df is None or df.empty or metric not in df.columns:
                continue

            v = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v) & (v > 0)]

            if len(v):
                vals.append(v)

    if not vals:
        return np.array([], dtype=float)

    return np.concatenate(vals)


def metric_ylim(level_data: dict, metric: str) -> tuple[float, float]:
    vals = collect_positive_values(level_data, metric)

    if vals.size == 0:
        return 1e-12, 1.0

    ymin = max(1e-14, float(np.min(vals)) * 0.75)
    ymax = float(np.max(vals)) * 1.35

    if ymin <= 0 or not np.isfinite(ymin):
        ymin = 1e-14

    if ymax <= ymin or not np.isfinite(ymax):
        ymax = ymin * 10.0

    return ymin, ymax


def add_column_headers(fig, axes) -> None:
    for col, (_, title) in enumerate(PANEL_DEFS):
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
            zorder=20,
        )


def add_metric_row_labels(fig, axes) -> None:
    row_labels = ["WAIT/OP", "VALU/OP"]

    for row, label in enumerate(row_labels):
        left_ax = axes[row, 0]
        pos = left_ax.get_position()

        width = 0.034
        x = pos.x0 - 0.068
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
            label,
            ha="center",
            va="center",
            rotation=90,
            fontsize=ROW_LABEL_FONTSIZE,
            fontweight="bold",
            zorder=21,
        )


def add_gpu_label(fig, gpu: str) -> None:
    fig.text(
        0.5,
        0.965,
        f"{norm_gpu_name(gpu)} Profiling Metrics Per Successful Operation",
        ha="center",
        va="center",
        fontsize=GPU_LABEL_FONTSIZE,
        fontweight="bold",
    )


# ============================================================
# Plot
# ============================================================

def plot_profile_2x4(gpu: str, out_path: Path) -> None:
    gpu = norm_gpu_name(gpu)

    level_data = {
        level: build_level_data(gpu, level)
        for level, _ in PANEL_DEFS
    }

    ylim_wait = metric_ylim(level_data, "Wait_per_op")
    ylim_valu = metric_ylim(level_data, "VALU_per_op")

    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.4))

    plt.subplots_adjust(
        left=0.13,
        right=0.992,
        bottom=0.12,
        top=0.82,
        wspace=0.045,
        hspace=0.09,
    )

    legend_handles = []
    legend_labels = []

    row_metric_map = {
        0: ("Wait_per_op", ylim_wait),
        1: ("VALU_per_op", ylim_valu),
    }

    for row in range(2):
        metric, ylim = row_metric_map[row]

        for col, (level, _) in enumerate(PANEL_DEFS):
            ax = axes[row, col]
            panel_data = level_data[level]

            ax.set_facecolor(SHADE_FACE if col in SHADED_COLUMNS else WHITE_FACE)

            all_threads = sorted({
                int(t)
                for queue in QUEUES
                for t in panel_data.get(queue, pd.DataFrame()).get(
                    "Grid_Size",
                    pd.Series(dtype=float),
                ).tolist()
                if pd.notna(t)
            })

            for queue in QUEUES:
                df = panel_data.get(queue)

                if df is None or df.empty or metric not in df.columns:
                    continue

                x = pd.to_numeric(df["Grid_Size"], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(df[metric], errors="coerce").to_numpy(dtype=float)

                keep = np.isfinite(x) & np.isfinite(y) & (y > 0)

                if not np.any(keep):
                    continue

                line, = ax.plot(
                    x[keep],
                    y[keep],
                    linestyle="-",
                    marker=QUEUE_MARKERS[queue],
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    markeredgewidth=MARKER_EDGE_WIDTH,
                    color=QUEUE_COLORS[queue],
                    label=QUEUE_LABELS[queue],
                    zorder=5,
                )

                if QUEUE_LABELS[queue] not in legend_labels:
                    legend_handles.append(line)
                    legend_labels.append(QUEUE_LABELS[queue])

            set_log2_x(ax, all_threads)

            ax.set_yscale("log")
            ax.set_ylim(ylim)

            ax.grid(True, which="major", linestyle="--", alpha=0.30)
            ax.grid(True, which="minor", linestyle=":", alpha=0.16)

            ax.tick_params(axis="x", labelsize=TICK_FONTSIZE)
            ax.tick_params(axis="y", labelsize=TICK_FONTSIZE)

            if col != 0:
                ax.tick_params(axis="y", which="both", left=False, labelleft=False)

            if row != 1:
                ax.tick_params(axis="x", which="both", labelbottom=False)

            for spine in ax.spines.values():
                spine.set_linewidth(1.0)
                spine.set_color("#555555")

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            frameon=True,
            framealpha=0.95,
            facecolor="#f9f9f9",
            edgecolor="#cfcfcf",
            fancybox=True,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=4,
            fontsize=LEGEND_FONTSIZE,
            handlelength=2.2,
            columnspacing=1.5,
        )

    fig.text(
        0.5,
        0.055,
        "Threads",
        ha="center",
        va="center",
        fontsize=AXIS_LABEL_FONTSIZE,
    )

    add_gpu_label(fig, gpu)
    add_column_headers(fig, axes)
    add_metric_row_labels(fig, axes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out_path}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fixed 2x4 profiling plots from results/profiling."
    )

    parser.add_argument(
        "--gpu",
        choices=["MI210", "MI300A", "mi210", "mi300a", "both"],
        default="both",
        help="GPU figure to generate. Default: both.",
    )

    args = parser.parse_args()

    if not BENCHMARK_CSV.exists():
        raise FileNotFoundError(f"Missing benchmark CSV: {BENCHMARK_CSV}")

    if not PROFILE_ROOT.exists():
        raise FileNotFoundError(f"Missing profile data directory: {PROFILE_ROOT}")

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if args.gpu == "both":
        targets = ["MI210", "MI300A"]
    else:
        targets = [norm_gpu_name(args.gpu)]

    for gpu in targets:
        out_path = FIGURE_DIR / f"{gpu_slug(gpu)}_profiling_compact_2x4.png"
        plot_profile_2x4(gpu, out_path)


if __name__ == "__main__":
    main()