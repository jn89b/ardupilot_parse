#!/usr/bin/env python3
"""
analyze_landings.py

Analyze cached landing descents produced by cache_landing_descents.py.

Default input:
    data/landing_cache/<label>/

Default output:
    data/landing_analysis/<label>/

The script:
  1. Recursively loads all cached expert descent files (.parquet or .csv.gz)
  2. Re-bins samples by altitude using a configurable bin width
  3. Computes per-descent statistics in each altitude bin
  4. Aggregates those statistics across descents with equal landing weighting
  5. Computes raw-sample statistics as a secondary diagnostic
  6. Generates statistical trend plots for each requested variable
  7. Generates overall distribution histograms
  8. Generates coverage plots showing how many landings contribute to each bin
  9. Caches the statistical tables as Parquet (or CSV fallback)

IMPORTANT STATISTICAL NOTE
--------------------------
The primary aggregate table is based on PER-DESCENT bin summaries, not all
individual samples pooled together. This prevents a landing with more samples
or more time in an altitude bin from receiving more statistical weight than
another landing.

Example:
    python analyze_landings.py

Custom altitude bins:
    python analyze_landings.py --bin-width 2.5

Specific variables:
    python analyze_landings.py \
        --variables airspeed_mps aoa_deg sideslip_deg descent_rate_mps

Outputs include:
    per_descent_bin_stats.parquet
    altitude_stats.parquet
    raw_sample_altitude_stats.parquet
    overall_stats.parquet
    analysis_config.json
    plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_VARIABLES = [
    "airspeed_mps",
    "aoa_deg",
    "sideslip_deg",
    "descent_rate_mps",
    "groundspeed_mps",
    "pitch_deg",
    "roll_deg",
    "pilot_pitch_centered",
    "pilot_roll_centered",
    "pilot_throttle_norm",
    "throttle_output",
    "q_radps",
    "p_radps",
]

DISPLAY_NAMES = {
    "airspeed_mps": "Airspeed",
    "aoa_deg": "Angle of attack",
    "sideslip_deg": "Sideslip",
    "descent_rate_mps": "Descent rate",
    "groundspeed_mps": "Groundspeed",
    "pitch_deg": "Pitch",
    "roll_deg": "Roll",
    "pilot_pitch_centered": "Pilot pitch input",
    "pilot_roll_centered": "Pilot roll input",
    "pilot_throttle_norm": "Pilot throttle input",
    "throttle_output": "Throttle output",
    "q_radps": "Pitch rate",
    "p_radps": "Roll rate",
    "r_radps": "Yaw rate",
    "pitch_error_deg": "Pitch error",
    "roll_error_deg": "Roll error",
}

UNITS = {
    "airspeed_mps": "m/s",
    "aoa_deg": "deg",
    "sideslip_deg": "deg",
    "descent_rate_mps": "m/s",
    "groundspeed_mps": "m/s",
    "pitch_deg": "deg",
    "roll_deg": "deg",
    "pilot_pitch_centered": "normalized stick",
    "pilot_roll_centered": "normalized stick",
    "pilot_throttle_norm": "normalized throttle",
    "throttle_output": "output",
    "q_radps": "rad/s",
    "p_radps": "rad/s",
    "r_radps": "rad/s",
    "pitch_error_deg": "deg",
    "roll_error_deg": "deg",
}


# -----------------------------------------------------------------------------
# I/O
# -----------------------------------------------------------------------------

def discover_cache_files(input_dir: Path) -> List[Path]:
    parquet = sorted(input_dir.rglob("*.parquet"))
    csv_gz = sorted(input_dir.rglob("*.csv.gz"))

    # Prefer parquet if both representations of the same stem happen to exist.
    parquet_stems = {p.name.removesuffix(".parquet") for p in parquet}
    filtered_csv = [
        p for p in csv_gz
        if p.name.removesuffix(".csv.gz") not in parquet_stems
    ]

    return parquet + filtered_csv


def load_one(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.name.lower().endswith(".csv.gz"):
        return pd.read_csv(path)

    raise ValueError(f"Unsupported cache file: {path}")


def load_dataset(input_dir: Path) -> pd.DataFrame:
    files = discover_cache_files(input_dir)

    if not files:
        raise FileNotFoundError(
            f"No .parquet or .csv.gz landing-cache files found under: {input_dir}"
        )

    frames: List[pd.DataFrame] = []

    for path in files:
        df = load_one(path)

        if df.empty:
            continue

        if "descent_id" not in df.columns:
            # Safe fallback for older cache files.
            df = df.copy()
            df["descent_id"] = path.name.split(".")[0]

        if "log_file" not in df.columns:
            df["log_file"] = path.name

        df["cache_source_file"] = str(path)
        frames.append(df)

    if not frames:
        raise ValueError("Cache files were found, but all were empty.")

    return pd.concat(frames, ignore_index=True, sort=False)


def write_table(df: pd.DataFrame, base_path: Path) -> Path:
    """
    Prefer Parquet. Fall back to compressed CSV when parquet support is absent.
    """
    base_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        csv_path = base_path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        return csv_path


# -----------------------------------------------------------------------------
# Altitude bins
# -----------------------------------------------------------------------------

def add_altitude_bins(
    df: pd.DataFrame,
    bin_width_m: float,
    min_altitude_m: Optional[float],
    max_altitude_m: Optional[float],
) -> pd.DataFrame:
    if "altitude_rel_m" not in df.columns:
        raise KeyError(
            "Expected 'altitude_rel_m' in cached landing data. "
            "Run cache_landing_descents.py first."
        )

    out = df.copy()
    out["altitude_rel_m"] = pd.to_numeric(
        out["altitude_rel_m"], errors="coerce"
    )

    out = out[np.isfinite(out["altitude_rel_m"])].copy()

    if min_altitude_m is not None:
        out = out[out["altitude_rel_m"] >= min_altitude_m]

    if max_altitude_m is not None:
        out = out[out["altitude_rel_m"] <= max_altitude_m]

    # Example for 5 m bins:
    #   altitude 27.3 -> lower=25, upper=30, center=27.5
    lower = np.floor(out["altitude_rel_m"] / bin_width_m) * bin_width_m
    upper = lower + bin_width_m

    out["altitude_bin_lower_m"] = lower
    out["altitude_bin_upper_m"] = upper
    out["altitude_bin_center_m"] = lower + bin_width_m / 2.0

    return out


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------

def finite_numeric(df: pd.DataFrame, variable: str) -> pd.Series:
    return pd.to_numeric(df[variable], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def compute_per_descent_bin_stats(
    df: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    """
    One row per:
        descent_id x altitude_bin x variable

    These rows are the units used for equal-landing-weight aggregation.
    """
    rows: List[Dict[str, object]] = []

    group_cols = [
        "descent_id",
        "altitude_bin_lower_m",
        "altitude_bin_upper_m",
        "altitude_bin_center_m",
    ]

    for keys, group in df.groupby(group_cols, observed=True, sort=True):
        descent_id, bin_low, bin_high, bin_center = keys

        log_file = (
            str(group["log_file"].iloc[0])
            if "log_file" in group.columns and len(group)
            else ""
        )

        for variable in variables:
            if variable not in group.columns:
                continue

            x = finite_numeric(group, variable).dropna()
            if x.empty:
                continue

            rows.append(
                {
                    "descent_id": descent_id,
                    "log_file": log_file,
                    "altitude_bin_lower_m": float(bin_low),
                    "altitude_bin_upper_m": float(bin_high),
                    "altitude_bin_center_m": float(bin_center),
                    "variable": variable,
                    "n_samples": int(len(x)),
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "median": float(x.median()),
                    "q05": float(x.quantile(0.05)),
                    "q25": float(x.quantile(0.25)),
                    "q75": float(x.quantile(0.75)),
                    "q95": float(x.quantile(0.95)),
                    "min": float(x.min()),
                    "max": float(x.max()),
                }
            )

    return pd.DataFrame(rows)


def compute_equal_landing_altitude_stats(
    per_descent: pd.DataFrame,
) -> pd.DataFrame:
    """
    Primary statistical table.

    For each altitude bin and variable, aggregate the PER-DESCENT MEANS.
    Every landing therefore contributes at most one number to a variable/bin.

    The between_landing_* fields describe variation from landing to landing.
    """
    if per_descent.empty:
        return pd.DataFrame()

    rows: List[Dict[str, object]] = []

    group_cols = [
        "altitude_bin_lower_m",
        "altitude_bin_upper_m",
        "altitude_bin_center_m",
        "variable",
    ]

    for keys, group in per_descent.groupby(group_cols, observed=True, sort=True):
        bin_low, bin_high, bin_center, variable = keys

        means = pd.to_numeric(group["mean"], errors="coerce").dropna()
        medians = pd.to_numeric(group["median"], errors="coerce").dropna()

        if means.empty:
            continue

        rows.append(
            {
                "altitude_bin_lower_m": float(bin_low),
                "altitude_bin_upper_m": float(bin_high),
                "altitude_bin_center_m": float(bin_center),
                "variable": variable,
                "n_landings": int(group["descent_id"].nunique()),
                "total_samples": int(group["n_samples"].sum()),
                # Equal-landing-weight estimate of central tendency.
                "mean": float(means.mean()),
                "std_between_landings": (
                    float(means.std(ddof=1)) if len(means) > 1 else np.nan
                ),
                "median": float(means.median()),
                "q05": float(means.quantile(0.05)),
                "q25": float(means.quantile(0.25)),
                "q75": float(means.quantile(0.75)),
                "q95": float(means.quantile(0.95)),
                "min_landing_mean": float(means.min()),
                "max_landing_mean": float(means.max()),
                "mean_of_landing_medians": (
                    float(medians.mean()) if not medians.empty else np.nan
                ),
            }
        )

    return pd.DataFrame(rows)


def compute_raw_sample_altitude_stats(
    df: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    """
    Secondary / diagnostic table based on all samples pooled together.

    Use this for understanding the raw sample distribution, but do NOT treat
    it as the primary across-landing inference because flights with more
    samples would receive more weight.
    """
    rows: List[Dict[str, object]] = []

    group_cols = [
        "altitude_bin_lower_m",
        "altitude_bin_upper_m",
        "altitude_bin_center_m",
    ]

    for keys, group in df.groupby(group_cols, observed=True, sort=True):
        bin_low, bin_high, bin_center = keys

        for variable in variables:
            if variable not in group.columns:
                continue

            x = finite_numeric(group, variable).dropna()
            if x.empty:
                continue

            rows.append(
                {
                    "altitude_bin_lower_m": float(bin_low),
                    "altitude_bin_upper_m": float(bin_high),
                    "altitude_bin_center_m": float(bin_center),
                    "variable": variable,
                    "n_samples": int(len(x)),
                    "n_landings": int(group.loc[x.index, "descent_id"].nunique()),
                    "mean": float(x.mean()),
                    "std": float(x.std(ddof=1)) if len(x) > 1 else np.nan,
                    "median": float(x.median()),
                    "q05": float(x.quantile(0.05)),
                    "q25": float(x.quantile(0.25)),
                    "q75": float(x.quantile(0.75)),
                    "q95": float(x.quantile(0.95)),
                    "min": float(x.min()),
                    "max": float(x.max()),
                }
            )

    return pd.DataFrame(rows)


def compute_overall_stats(
    per_descent: pd.DataFrame,
    variables: Sequence[str],
) -> pd.DataFrame:
    """
    One overall row per variable based on per-descent means.

    This gives a compact summary table that still gives every descent equal
    weight.
    """
    rows: List[Dict[str, object]] = []

    for variable in variables:
        sub = per_descent[per_descent["variable"] == variable]
        if sub.empty:
            continue

        # First collapse all altitude bins within each descent.
        landing_values = (
            sub.groupby("descent_id", observed=True)["mean"]
            .mean()
            .dropna()
        )

        if landing_values.empty:
            continue

        rows.append(
            {
                "variable": variable,
                "n_landings": int(len(landing_values)),
                "mean": float(landing_values.mean()),
                "std_between_landings": (
                    float(landing_values.std(ddof=1))
                    if len(landing_values) > 1
                    else np.nan
                ),
                "median": float(landing_values.median()),
                "q05": float(landing_values.quantile(0.05)),
                "q25": float(landing_values.quantile(0.25)),
                "q75": float(landing_values.quantile(0.75)),
                "q95": float(landing_values.quantile(0.95)),
                "min": float(landing_values.min()),
                "max": float(landing_values.max()),
            }
        )

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------

def pretty_name(variable: str) -> str:
    return DISPLAY_NAMES.get(variable, variable.replace("_", " ").title())


def variable_unit(variable: str) -> str:
    return UNITS.get(variable, "")


def save_altitude_stat_plot(
    altitude_stats: pd.DataFrame,
    variable: str,
    out_dir: Path,
    reverse_altitude_axis: bool,
    label: str,
) -> Optional[Path]:
    sub = altitude_stats[
        altitude_stats["variable"] == variable
    ].sort_values("altitude_bin_center_m")

    if sub.empty:
        return None

    x = sub["altitude_bin_center_m"].to_numpy(dtype=float)
    mean = sub["mean"].to_numpy(dtype=float)
    q25 = sub["q25"].to_numpy(dtype=float)
    q75 = sub["q75"].to_numpy(dtype=float)
    q05 = sub["q05"].to_numpy(dtype=float)
    q95 = sub["q95"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Wide and narrow distribution envelopes.
    ax.fill_between(x, q05, q95, alpha=0.15, label="5th–95th percentile")
    ax.fill_between(x, q25, q75, alpha=0.28, label="25th–75th percentile")
    ax.plot(x, mean, linewidth=2.0, label="Mean across landings")

    ax.set_title(f"{pretty_name(variable)} vs altitude — {label} landings")
    ax.set_xlabel("Relative altitude (m)")

    unit = variable_unit(variable)
    ylabel = pretty_name(variable)
    if unit:
        ylabel += f" ({unit})"
    ax.set_ylabel(ylabel)

    ax.grid(True, alpha=0.25)
    ax.legend()
    if reverse_altitude_axis:
        ax.invert_xaxis()

    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variable}_vs_altitude.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_overall_histogram(
    df: pd.DataFrame,
    variable: str,
    out_dir: Path,
    bins: int,
    label: str,
) -> Optional[Path]:
    if variable not in df.columns:
        return None

    x = finite_numeric(df, variable).dropna()
    if x.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(x.to_numpy(dtype=float), bins=bins)

    ax.axvline(float(x.mean()), linestyle="--", linewidth=1.5, label="Mean")
    ax.axvline(float(x.median()), linestyle=":", linewidth=1.5, label="Median")

    ax.set_title(f"{pretty_name(variable)} distribution — {label} samples")

    unit = variable_unit(variable)
    xlabel = pretty_name(variable)
    if unit:
        xlabel += f" ({unit})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Samples")

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variable}_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_landing_mean_histogram(
    per_descent: pd.DataFrame,
    variable: str,
    out_dir: Path,
    bins: int,
    label: str,
) -> Optional[Path]:
    """
    Distribution of landing-level means. This is usually more useful than the
    raw-sample histogram when you want to understand flight-to-flight variance.
    """
    sub = per_descent[per_descent["variable"] == variable]
    if sub.empty:
        return None

    values = (
        sub.groupby("descent_id", observed=True)["mean"]
        .mean()
        .dropna()
    )
    if values.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(values.to_numpy(dtype=float), bins=min(bins, max(5, len(values))))

    ax.axvline(
        float(values.mean()),
        linestyle="--",
        linewidth=1.5,
        label="Mean",
    )
    ax.axvline(
        float(values.median()),
        linestyle=":",
        linewidth=1.5,
        label="Median",
    )

    ax.set_title(f"{pretty_name(variable)} — distribution across {label} landings")

    unit = variable_unit(variable)
    xlabel = f"Landing mean {pretty_name(variable).lower()}"
    if unit:
        xlabel += f" ({unit})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Landings")

    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variable}_landing_means_histogram.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_coverage_plot(
    altitude_stats: pd.DataFrame,
    out_dir: Path,
    reverse_altitude_axis: bool,
    label: str,
) -> Optional[Path]:
    if altitude_stats.empty:
        return None

    coverage = (
        altitude_stats.groupby("altitude_bin_center_m", observed=True)["n_landings"]
        .max()
        .reset_index()
        .sort_values("altitude_bin_center_m")
    )

    if coverage.empty:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        coverage["altitude_bin_center_m"],
        coverage["n_landings"],
        marker="o",
    )
    ax.set_title(f"{label.title()} landing coverage by altitude")
    ax.set_xlabel("Relative altitude (m)")
    ax.set_ylabel("Number of contributing landings")
    ax.grid(True, alpha=0.25)

    if reverse_altitude_axis:
        ax.invert_xaxis()

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "landing_coverage_vs_altitude.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def save_correlation_matrix(
    df: pd.DataFrame,
    variables: Sequence[str],
    out_dir: Path,
    label: str,
) -> Optional[Path]:
    available = [
        v for v in variables
        if v in df.columns and pd.to_numeric(df[v], errors="coerce").notna().any()
    ]

    if len(available) < 2:
        return None

    numeric = df[available].apply(pd.to_numeric, errors="coerce")
    corr = numeric.corr()

    fig, ax = plt.subplots(
        figsize=(max(8, 0.8 * len(available)), max(7, 0.7 * len(available)))
    )

    im = ax.imshow(corr.to_numpy(dtype=float), vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(available)))
    ax.set_yticks(np.arange(len(available)))
    ax.set_xticklabels([pretty_name(v) for v in available], rotation=45, ha="right")
    ax.set_yticklabels([pretty_name(v) for v in available])

    for i in range(len(available)):
        for j in range(len(available)):
            val = corr.iloc[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    ax.set_title(f"{label.title()} landing sample correlation matrix")
    fig.colorbar(im, ax=ax, label="Pearson correlation")
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "correlation_matrix.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


# -----------------------------------------------------------------------------
# Main analysis
# -----------------------------------------------------------------------------

def analyze(
    input_dir: Path,
    output_dir: Path,
    label: str,
    variables: Sequence[str],
    bin_width_m: float,
    min_altitude_m: Optional[float],
    max_altitude_m: Optional[float],
    histogram_bins: int,
    reverse_altitude_axis: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"

    print(f"Loading {label} landing cache from: {input_dir}")
    df = load_dataset(input_dir)

    print(f"Loaded {len(df):,} samples")
    print(f"Detected {df['descent_id'].nunique()} unique descent(s)")

    available = [v for v in variables if v in df.columns]
    missing = [v for v in variables if v not in df.columns]

    if missing:
        print("\nVariables not present and skipped:")
        for v in missing:
            print(f"  - {v}")

    if not available:
        raise ValueError(
            "None of the requested variables exist in the cached dataset."
        )

    print("\nAnalyzing variables:")
    for v in available:
        print(f"  - {v}")

    binned = add_altitude_bins(
        df=df,
        bin_width_m=bin_width_m,
        min_altitude_m=min_altitude_m,
        max_altitude_m=max_altitude_m,
    )

    per_descent = compute_per_descent_bin_stats(
        binned,
        variables=available,
    )

    altitude_stats = compute_equal_landing_altitude_stats(per_descent)

    raw_stats = compute_raw_sample_altitude_stats(
        binned,
        variables=available,
    )

    overall_stats = compute_overall_stats(
        per_descent,
        variables=available,
    )

    # Cache tables.
    table_paths = [
        write_table(
            per_descent,
            output_dir / "per_descent_bin_stats",
        ),
        write_table(
            altitude_stats,
            output_dir / "altitude_stats",
        ),
        write_table(
            raw_stats,
            output_dir / "raw_sample_altitude_stats",
        ),
        write_table(
            overall_stats,
            output_dir / "overall_stats",
        ),
    ]

    # Also make small CSV copies of the main summary tables because they are
    # convenient to inspect in a text editor / spreadsheet.
    altitude_stats.to_csv(
        output_dir / "altitude_stats.csv",
        index=False,
    )
    overall_stats.to_csv(
        output_dir / "overall_stats.csv",
        index=False,
    )

    config = {
        "label": label,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "variables_requested": list(variables),
        "variables_analyzed": available,
        "variables_missing": missing,
        "bin_width_m": bin_width_m,
        "min_altitude_m": min_altitude_m,
        "max_altitude_m": max_altitude_m,
        "histogram_bins": histogram_bins,
        "reverse_altitude_axis": reverse_altitude_axis,
        "n_samples": int(len(binned)),
        "n_descents": int(binned["descent_id"].nunique()),
    }

    (output_dir / "analysis_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    # Plots.
    generated_plots: List[Path] = []

    coverage = save_coverage_plot(
        altitude_stats,
        plots_dir,
        reverse_altitude_axis=reverse_altitude_axis,
        label=label,
    )
    if coverage:
        generated_plots.append(coverage)

    corr = save_correlation_matrix(
        binned,
        available,
        plots_dir,
        label=label,
    )
    if corr:
        generated_plots.append(corr)

    for variable in available:
        p = save_altitude_stat_plot(
            altitude_stats,
            variable,
            plots_dir,
            reverse_altitude_axis=reverse_altitude_axis,
            label=label,
        )
        if p:
            generated_plots.append(p)

        p = save_overall_histogram(
            binned,
            variable,
            plots_dir,
            bins=histogram_bins,
            label=label,
        )
        if p:
            generated_plots.append(p)

        p = save_landing_mean_histogram(
            per_descent,
            variable,
            plots_dir,
            bins=histogram_bins,
            label=label,
        )
        if p:
            generated_plots.append(p)

    print("\nCached statistical tables:")
    for p in table_paths:
        print(f"  {p}")

    print(f"\nGenerated {len(generated_plots)} plot(s) under:")
    print(f"  {plots_dir}")

    print(f"\nPrimary table for {label} landing envelope:")
    print(f"  {output_dir / 'altitude_stats.csv'}")
    print(
        "\nUse altitude_stats for comparisons against failed landings. "
        "Its percentiles are based on landing-level bin means, giving each "
        "landing equal weight."
    )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate statistical tables and plots from cached landing data."
    )

    parser.add_argument(
        "--label",
        type=str,
        default="expert",
        help=(
            "Landing class/directory to analyze, e.g. expert or fail "
            "(default: expert)"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help=(
            "Optional cache directory override. Default: "
            "data/landing_cache/<label>"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Optional output directory override. Default: "
            "data/landing_analysis/<label>"
        ),
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=DEFAULT_VARIABLES,
        help="Variables to analyze",
    )
    parser.add_argument(
        "--bin-width",
        type=float,
        default=5.0,
        help="Altitude-bin width in meters (default: 5)",
    )
    parser.add_argument(
        "--min-altitude",
        type=float,
        default=0.0,
        help="Minimum relative altitude included in analysis",
    )
    parser.add_argument(
        "--max-altitude",
        type=float,
        default=100.0,
        help="Maximum relative altitude included in analysis",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=40,
        help="Number of bins for overall histograms",
    )
    parser.add_argument(
        "--ascending-altitude-axis",
        action="store_true",
        help=(
            "Plot altitude increasing left-to-right. Default is reversed so "
            "plots visually progress from high altitude toward touchdown."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.bin_width <= 0:
        raise ValueError("--bin-width must be > 0")

    label = args.label.strip()
    if not label:
        raise ValueError("--label cannot be empty")

    input_dir = (
        args.input
        if args.input is not None
        else Path("data/landing_cache") / label
    )
    output_dir = (
        args.output
        if args.output is not None
        else Path("data/landing_analysis") / label
    )

    analyze(
        input_dir=input_dir,
        output_dir=output_dir,
        label=label,
        variables=args.variables,
        bin_width_m=args.bin_width,
        min_altitude_m=args.min_altitude,
        max_altitude_m=args.max_altitude,
        histogram_bins=args.histogram_bins,
        reverse_altitude_axis=not args.ascending_altitude_axis,
    )


if __name__ == "__main__":
    main()
