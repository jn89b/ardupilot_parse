#!/usr/bin/env python3
"""
compare_landing_results.py

Compare two altitude-binned landing statistics files, typically:

    data/landing_analysis/expert/altitude_stats.csv
    data/landing_analysis/fail/altitude_stats.csv

The script produces:
  - altitude_comparison.csv
  - discussion_summary.csv
  - discussion_report.md
  - overlay plots for each variable
  - difference plots (comparison - reference) for each variable

The "reference" dataset should normally be your expert/good FBWA data.
The "comparison" dataset can be AUTO, fail, or any other dataset.

Example:
    uv run compare_landing_results.py \
        --reference data/landing_analysis/expert/altitude_stats.csv \
        --comparison data/landing_analysis/fail/altitude_stats.csv \
        --reference-label expert_fbwa \
        --comparison-label fail_auto \
        --min-altitude 0 \
        --max-altitude 40

You can also limit variables:
    uv run compare_landing_results.py \
        --reference altitude_stats_expert.csv \
        --comparison altitude_stats_fail.csv \
        --variables aoa_deg airspeed_mps descent_rate_mps pitch_deg
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_VARIABLES = [
    "aoa_deg",
    "airspeed_mps",
    "descent_rate_mps",
    "pitch_deg",
    "sideslip_deg",
    "groundspeed_mps",
    "pilot_pitch_centered",
    "pilot_roll_centered",
    "pilot_throttle_norm",
    "throttle_output",
    "q_radps",
    "p_radps",
]


def safe_pct_diff(comparison: pd.Series, reference: pd.Series) -> pd.Series:
    """Percent difference relative to reference, avoiding divide-by-near-zero."""
    ref = pd.to_numeric(reference, errors="coerce")
    comp = pd.to_numeric(comparison, errors="coerce")
    out = pd.Series(np.nan, index=ref.index, dtype=float)

    mask = ref.abs() > 1e-9
    out.loc[mask] = 100.0 * (comp.loc[mask] - ref.loc[mask]) / ref.loc[mask]
    return out


def load_stats(path: Path, min_alt: float | None, max_alt: float | None) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = {
        "altitude_bin_lower_m",
        "altitude_bin_upper_m",
        "altitude_bin_center_m",
        "variable",
        "n_landings",
        "mean",
        "median",
        "q05",
        "q25",
        "q75",
        "q95",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}"
        )

    df = df.copy()

    if min_alt is not None:
        df = df[df["altitude_bin_center_m"] >= min_alt]
    if max_alt is not None:
        df = df[df["altitude_bin_center_m"] <= max_alt]

    return df


def merge_stats(
    ref: pd.DataFrame,
    comp: pd.DataFrame,
    ref_label: str,
    comp_label: str,
) -> pd.DataFrame:
    keys = [
        "altitude_bin_lower_m",
        "altitude_bin_upper_m",
        "altitude_bin_center_m",
        "variable",
    ]

    keep = keys + [
        "n_landings",
        "total_samples",
        "mean",
        "std_between_landings",
        "median",
        "q05",
        "q25",
        "q75",
        "q95",
        "min_landing_mean",
        "max_landing_mean",
        "mean_of_landing_medians",
    ]

    keep_ref = [c for c in keep if c in ref.columns]
    keep_comp = [c for c in keep if c in comp.columns]

    ref2 = ref[keep_ref].copy()
    comp2 = comp[keep_comp].copy()

    ref2 = ref2.rename(
        columns={c: f"{ref_label}_{c}" for c in ref2.columns if c not in keys}
    )
    comp2 = comp2.rename(
        columns={c: f"{comp_label}_{c}" for c in comp2.columns if c not in keys}
    )

    merged = ref2.merge(comp2, on=keys, how="inner")

    ref_mean = merged[f"{ref_label}_mean"]
    comp_mean = merged[f"{comp_label}_mean"]

    merged["mean_diff"] = comp_mean - ref_mean
    merged["abs_mean_diff"] = merged["mean_diff"].abs()
    merged["mean_pct_diff"] = safe_pct_diff(comp_mean, ref_mean)

    merged["median_diff"] = (
        merged[f"{comp_label}_median"] - merged[f"{ref_label}_median"]
    )

    # Whether the comparison mean sits outside the expert/reference IQR or 5-95 band.
    merged["comparison_mean_outside_reference_iqr"] = (
        (comp_mean < merged[f"{ref_label}_q25"])
        | (comp_mean > merged[f"{ref_label}_q75"])
    )

    merged["comparison_mean_outside_reference_90pct"] = (
        (comp_mean < merged[f"{ref_label}_q05"])
        | (comp_mean > merged[f"{ref_label}_q95"])
    )

    # Normalize the mean difference by between-landing std when available.
    std_col = f"{ref_label}_std_between_landings"
    if std_col in merged.columns:
        std = pd.to_numeric(merged[std_col], errors="coerce")
        merged["diff_in_reference_std"] = np.where(
            std.abs() > 1e-9,
            merged["mean_diff"] / std,
            np.nan,
        )

    return merged


def make_summary(
    merged: pd.DataFrame,
    ref_label: str,
    comp_label: str,
) -> pd.DataFrame:
    rows = []

    for variable, g in merged.groupby("variable", sort=True):
        g = g.sort_values("altitude_bin_center_m").copy()

        valid = g["mean_diff"].notna()
        if not valid.any():
            continue

        gv = g.loc[valid]
        max_idx = gv["abs_mean_diff"].idxmax()
        max_row = gv.loc[max_idx]

        outside_iqr_fraction = float(
            gv["comparison_mean_outside_reference_iqr"].mean()
        )
        outside_90_fraction = float(
            gv["comparison_mean_outside_reference_90pct"].mean()
        )

        rows.append(
            {
                "variable": variable,
                "n_altitude_bins_compared": int(len(gv)),
                f"{ref_label}_mean_across_bins": float(
                    gv[f"{ref_label}_mean"].mean()
                ),
                f"{comp_label}_mean_across_bins": float(
                    gv[f"{comp_label}_mean"].mean()
                ),
                "mean_difference_across_bins": float(gv["mean_diff"].mean()),
                "median_difference_across_bins": float(gv["mean_diff"].median()),
                "mean_absolute_difference_across_bins": float(
                    gv["abs_mean_diff"].mean()
                ),
                "max_absolute_difference": float(max_row["abs_mean_diff"]),
                "altitude_of_max_difference_m": float(
                    max_row["altitude_bin_center_m"]
                ),
                "signed_difference_at_max": float(max_row["mean_diff"]),
                "fraction_bins_outside_reference_iqr": outside_iqr_fraction,
                "fraction_bins_outside_reference_90pct": outside_90_fraction,
            }
        )

    return pd.DataFrame(rows)


def plot_overlay(
    merged: pd.DataFrame,
    variable: str,
    ref_label: str,
    comp_label: str,
    out_path: Path,
) -> None:
    g = merged[merged["variable"] == variable].sort_values(
        "altitude_bin_center_m"
    )
    if g.empty:
        return

    x = g["altitude_bin_center_m"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Reference/expert envelope
    ax.fill_between(
        x,
        g[f"{ref_label}_q05"].to_numpy(),
        g[f"{ref_label}_q95"].to_numpy(),
        alpha=0.12,
        label=f"{ref_label} 5–95%",
    )
    ax.fill_between(
        x,
        g[f"{ref_label}_q25"].to_numpy(),
        g[f"{ref_label}_q75"].to_numpy(),
        alpha=0.22,
        label=f"{ref_label} 25–75%",
    )

    ax.plot(
        x,
        g[f"{ref_label}_mean"].to_numpy(),
        marker="o",
        linewidth=2,
        label=f"{ref_label} mean",
    )
    ax.plot(
        x,
        g[f"{comp_label}_mean"].to_numpy(),
        marker="o",
        linewidth=2,
        label=f"{comp_label} mean",
    )

    ax.set_title(f"{variable}: {ref_label} vs {comp_label}")
    ax.set_xlabel("Altitude AGL (m)")
    ax.set_ylabel(variable)
    ax.grid(True, alpha=0.25)
    ax.legend()

    # Show high altitude on left, touchdown on right.
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_difference(
    merged: pd.DataFrame,
    variable: str,
    ref_label: str,
    comp_label: str,
    out_path: Path,
) -> None:
    g = merged[merged["variable"] == variable].sort_values(
        "altitude_bin_center_m"
    )
    if g.empty:
        return

    x = g["altitude_bin_center_m"].to_numpy()
    y = g["mean_diff"].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0.0, linewidth=1)
    ax.plot(x, y, marker="o", linewidth=2)

    ax.set_title(
        f"{variable}: mean difference ({comp_label} - {ref_label})"
    )
    ax.set_xlabel("Altitude AGL (m)")
    ax.set_ylabel(f"Δ {variable}")
    ax.grid(True, alpha=0.25)
    ax.invert_xaxis()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def fmt(v: float, digits: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def write_markdown_report(
    summary: pd.DataFrame,
    merged: pd.DataFrame,
    variables: list[str],
    ref_label: str,
    comp_label: str,
    path: Path,
) -> None:
    lines = [
        "# Landing Dataset Comparison",
        "",
        f"**Reference:** `{ref_label}`  ",
        f"**Comparison:** `{comp_label}`",
        "",
        "This report is descriptive. It shows where the comparison dataset differs "
        "from the reference/expert envelope; it does not establish causation.",
        "",
    ]

    for variable in variables:
        s = summary[summary["variable"] == variable]
        g = merged[merged["variable"] == variable].sort_values(
            "altitude_bin_center_m",
            ascending=False,
        )
        if s.empty or g.empty:
            continue

        row = s.iloc[0]

        lines.extend(
            [
                f"## {variable}",
                "",
                f"- Mean difference across altitude bins "
                f"(`{comp_label} - {ref_label}`): "
                f"**{fmt(row['mean_difference_across_bins'])}**",
                f"- Mean absolute difference across bins: "
                f"**{fmt(row['mean_absolute_difference_across_bins'])}**",
                f"- Largest separation occurs around "
                f"**{fmt(row['altitude_of_max_difference_m'], 1)} m AGL**, "
                f"with a signed difference of "
                f"**{fmt(row['signed_difference_at_max'])}**.",
                f"- Comparison mean is outside the reference 25–75% envelope in "
                f"**{100.0 * row['fraction_bins_outside_reference_iqr']:.0f}%** "
                f"of compared altitude bins.",
                f"- Comparison mean is outside the reference 5–95% envelope in "
                f"**{100.0 * row['fraction_bins_outside_reference_90pct']:.0f}%** "
                f"of compared altitude bins.",
                "",
                "| AGL (m) | "
                f"{ref_label} mean | "
                f"{comp_label} mean | Δ mean | "
                f"{ref_label} q25–q75 |",
                "|---:|---:|---:|---:|---:|",
            ]
        )

        # Keep the markdown readable: use every available bin for <= 40 bins.
        for _, r in g.iterrows():
            lines.append(
                f"| {r['altitude_bin_center_m']:.1f} "
                f"| {fmt(r[f'{ref_label}_mean'])} "
                f"| {fmt(r[f'{comp_label}_mean'])} "
                f"| {fmt(r['mean_diff'])} "
                f"| {fmt(r[f'{ref_label}_q25'])}–{fmt(r[f'{ref_label}_q75'])} |"
            )

        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two altitude-binned landing analysis CSV files."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        required=True,
        help="Reference/expert altitude_stats.csv",
    )
    parser.add_argument(
        "--comparison",
        type=Path,
        required=True,
        help="Comparison altitude_stats.csv",
    )
    parser.add_argument(
        "--reference-label",
        default="expert",
        help="Short label used in output columns and plots.",
    )
    parser.add_argument(
        "--comparison-label",
        default="comparison",
        help="Short label used in output columns and plots.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/landing_analysis/comparison"),
        help="Output directory.",
    )
    parser.add_argument(
        "--min-altitude",
        type=float,
        default=None,
        help="Minimum altitude-bin center to include.",
    )
    parser.add_argument(
        "--max-altitude",
        type=float,
        default=None,
        help="Maximum altitude-bin center to include.",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=None,
        help="Variables to compare. Default: common landing variables.",
    )

    args = parser.parse_args()

    ref = load_stats(args.reference, args.min_altitude, args.max_altitude)
    comp = load_stats(args.comparison, args.min_altitude, args.max_altitude)

    common_vars = sorted(set(ref["variable"]) & set(comp["variable"]))

    if args.variables:
        variables = [v for v in args.variables if v in common_vars]
        missing = [v for v in args.variables if v not in common_vars]
        if missing:
            print(f"Skipping missing variable(s): {', '.join(missing)}")
    else:
        variables = [v for v in DEFAULT_VARIABLES if v in common_vars]

    if not variables:
        raise RuntimeError("No requested variables exist in both input files.")

    ref = ref[ref["variable"].isin(variables)].copy()
    comp = comp[comp["variable"].isin(variables)].copy()

    merged = merge_stats(
        ref,
        comp,
        args.reference_label,
        args.comparison_label,
    )

    if merged.empty:
        raise RuntimeError(
            "No matching altitude bins / variables were found between the files."
        )

    summary = make_summary(
        merged,
        args.reference_label,
        args.comparison_label,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(args.output / "altitude_comparison.csv", index=False)
    summary.to_csv(args.output / "discussion_summary.csv", index=False)

    for variable in variables:
        plot_overlay(
            merged,
            variable,
            args.reference_label,
            args.comparison_label,
            plots_dir / f"{variable}_overlay.png",
        )
        plot_difference(
            merged,
            variable,
            args.reference_label,
            args.comparison_label,
            plots_dir / f"{variable}_difference.png",
        )

    write_markdown_report(
        summary,
        merged,
        variables,
        args.reference_label,
        args.comparison_label,
        args.output / "discussion_report.md",
    )

    print()
    print("Comparison complete.")
    print(f"  Detailed comparison : {args.output / 'altitude_comparison.csv'}")
    print(f"  Discussion summary  : {args.output / 'discussion_summary.csv'}")
    print(f"  Markdown report     : {args.output / 'discussion_report.md'}")
    print(f"  Plots               : {plots_dir}")
    print()
    print("Variables:")
    for v in variables:
        print(f"  - {v}")


if __name__ == "__main__":
    main()
