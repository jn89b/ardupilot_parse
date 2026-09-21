#!/usr/bin/env python3
"""
cache_landing_descents.py

Crawl:
    data/landings/
        expert/**/*.BIN
        fail/**/*.BIN

Extract low-altitude descent/landing segments from ArduPilot DataFlash logs and
cache them as analysis-ready tabular files.

The cache preserves:
  - label / source log / descent id
  - relative altitude and descent rate
  - GPS and EKF position/velocity
  - measured / estimated airspeed
  - attitude and desired attitude
  - body rates and accelerations
  - pilot RC inputs (RCIN)
  - servo outputs (RCOU)
  - Plane CTUN / NTUN / TECS fields when available
  - wind estimate when available
  - logged ArduPilot AOA / SSA when available
  - independently reconstructed AoA and sideslip for validation/fallback

AoA / sideslip note:
  The primary `aoa_deg` and `sideslip_deg` columns use the ArduPilot `AOA`
  message directly:
      AOA.AOA -> angle of attack [deg]
      AOA.SSA -> sideslip angle [deg]

  The script also preserves independently reconstructed values in
  `aoa_derived_deg` and `sideslip_derived_deg`. If the logged AOA/SSA message
  is unavailable for a row, the primary column falls back to the reconstructed
  value and records that fallback in the corresponding source column.

Outputs:
    data/landing_cache/
        expert/<log>__descent_000.parquet
        fail/<log>__descent_000.parquet
        manifest.csv

If parquet support (pyarrow/fastparquet) is unavailable, the script
automatically falls back to .csv.gz.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pymavlink import DFReader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MESSAGE_TYPES = [
    "GPS",
    "POS",
    "ATT",
    "IMU",
    "AHR2",
    "XKF0",
    "XKF1",
    "XKF2",
    "CTUN",
    "NTUN",
    "TECS",
    "ARSP",
    "AOA",
    "DCM",
    "RCIN",
    "RCOU",
    "BARO",
    "RFND",
    "MODE",
]

# Typical fixed-wing RC mapping. Override from CLI if your aircraft differs.
DEFAULT_ROLL_CH = 1
DEFAULT_PITCH_CH = 2
DEFAULT_THROTTLE_CH = 3
DEFAULT_YAW_CH = 4


@dataclass
class ExtractionConfig:
    dt: float = 0.05
    max_altitude_m: float = 120.0
    min_altitude_m: float = -10.0
    min_descent_rate_mps: float = 0.25
    min_duration_s: float = 5.0
    min_altitude_drop_m: float = 8.0
    merge_gap_s: float = 5.0
    pad_before_s: float = 2.0
    pad_after_s: float = 5.0
    smooth_altitude_s: float = 1.0
    final_only: bool = False
    roll_channel: int = DEFAULT_ROLL_CH
    pitch_channel: int = DEFAULT_PITCH_CH
    throttle_channel: int = DEFAULT_THROTTLE_CH
    yaw_channel: int = DEFAULT_YAW_CH


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def first_existing(df: pd.DataFrame, names: Sequence[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


def get_col(
    df: pd.DataFrame,
    names: Sequence[str],
    default: float = np.nan,
) -> np.ndarray:
    name = first_existing(df, names)
    if name is None:
        return np.full(len(df), default, dtype=float)
    return pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)


def wrap_pi(rad: np.ndarray) -> np.ndarray:
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


def rolling_smooth(x: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return x.astype(float, copy=True)
    s = pd.Series(x, dtype=float)
    # median is robust against occasional GPS / estimator spikes
    return (
        s.rolling(window_samples, center=True, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def fill_short_false_gaps(mask: np.ndarray, max_gap_samples: int) -> np.ndarray:
    """Close short False gaps bounded by True regions."""
    out = mask.astype(bool, copy=True)
    if max_gap_samples <= 0 or len(out) == 0:
        return out

    i = 0
    n = len(out)
    while i < n:
        if out[i]:
            i += 1
            continue

        start = i
        while i < n and not out[i]:
            i += 1
        end = i  # first True after gap, or n

        gap_len = end - start
        bounded_left = start > 0 and out[start - 1]
        bounded_right = end < n and out[end]

        if bounded_left and bounded_right and gap_len <= max_gap_samples:
            out[start:end] = True

    return out


def contiguous_true_regions(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive [start, end] regions."""
    regions: List[Tuple[int, int]] = []
    n = len(mask)
    i = 0

    while i < n:
        if not mask[i]:
            i += 1
            continue

        start = i
        while i + 1 < n and mask[i + 1]:
            i += 1
        regions.append((start, i))
        i += 1

    return regions


def choose_primary_instance(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prefer the active/primary sensor instance when fields exist.
    Safe for GPS, ARSP, IMU, BARO, etc.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Explicit "in use" flag.
    if "U" in out.columns:
        used = pd.to_numeric(out["U"], errors="coerce") == 1
        if used.any():
            out = out.loc[used].copy()

    # ARSP can identify its primary instance directly.
    if "I" in out.columns and "Pri" in out.columns:
        i = pd.to_numeric(out["I"], errors="coerce")
        pri = pd.to_numeric(out["Pri"], errors="coerce")
        primary = i == pri
        if primary.any():
            out = out.loc[primary].copy()

    # Otherwise prefer instance 0 when multiple instances exist.
    if "I" in out.columns:
        inst = pd.to_numeric(out["I"], errors="coerce")
        if (inst == 0).any():
            out = out.loc[inst == 0].copy()

    return out.sort_values("t").drop_duplicates("t", keep="last")


def align_to_grid(
    df: pd.DataFrame,
    t_grid: pd.DataFrame,
    tolerance_s: float,
) -> pd.DataFrame:
    if df is None or df.empty or "t" not in df.columns:
        return t_grid.copy()

    src = df.sort_values("t").drop_duplicates("t", keep="last").copy()
    return pd.merge_asof(
        t_grid.sort_values("t"),
        src,
        on="t",
        direction="nearest",
        tolerance=tolerance_s,
    )


def prefixed_aligned(
    df: pd.DataFrame,
    grid: pd.DataFrame,
    prefix: str,
    tolerance_s: float,
) -> pd.DataFrame:
    """
    Align a DataFlash message to the requested time grid and prefix every
    original field name so nothing useful gets silently discarded.
    """
    if df is None or df.empty:
        return grid[["t"]].copy()

    aligned = align_to_grid(df, grid, tolerance_s=tolerance_s)
    drop_cols = {"TimeUS", "_t", "mavpackettype"}
    rename: Dict[str, str] = {}

    for c in aligned.columns:
        if c == "t" or c in drop_cols:
            continue
        rename[c] = f"{prefix}_{c}"

    aligned = aligned.drop(
        columns=[c for c in drop_cols if c in aligned.columns],
        errors="ignore",
    )
    return aligned.rename(columns=rename)


# ---------------------------------------------------------------------------
# DataFlash reader
# ---------------------------------------------------------------------------

class FlightParser:
    def __init__(self, log_path: Path, verbose: bool = False) -> None:
        self.log_path = Path(log_path)
        self.binary_log = DFReader.DFReader_binary(filename=str(self.log_path))

        if verbose:
            names = sorted(fmt.name for fmt in self.binary_log.formats.values())
            print(f"[{self.log_path.name}] messages: {names}")

        self.binary_log.rewind()

    def read_messages(self, types: Sequence[str]) -> Dict[str, pd.DataFrame]:
        wanted = set(types)
        rows: Dict[str, List[Dict[str, Any]]] = {t: [] for t in types}

        while True:
            m = self.binary_log.recv_msg()
            if m is None:
                break

            mt = m.get_type()
            if mt not in wanted:
                continue

            d = m.to_dict()
            d["TimeUS"] = getattr(m, "TimeUS", d.get("TimeUS"))
            d["_t"] = getattr(m, "_timestamp", None)
            rows[mt].append(d)

        dfs = {name: pd.DataFrame(values) for name, values in rows.items()}

        all_timeus: List[pd.Series] = []
        for df in dfs.values():
            if not df.empty and "TimeUS" in df.columns:
                s = pd.to_numeric(df["TimeUS"], errors="coerce").dropna()
                if not s.empty:
                    all_timeus.append(s)

        if not all_timeus:
            raise ValueError("No TimeUS found in requested log messages.")

        t0_us = float(pd.concat(all_timeus, ignore_index=True).min())

        for name, df in dfs.items():
            if df.empty or "TimeUS" not in df.columns:
                continue

            tmp = df.copy()
            tmp["TimeUS"] = pd.to_numeric(tmp["TimeUS"], errors="coerce")
            tmp = tmp.dropna(subset=["TimeUS"]).sort_values("TimeUS")
            tmp["t"] = (tmp["TimeUS"] - t0_us) / 1e6
            dfs[name] = tmp

        # Prefer primary instances where that is meaningful.
        for name in ("GPS", "ARSP", "IMU", "BARO"):
            dfs[name] = choose_primary_instance(dfs.get(name, pd.DataFrame()))

        return dfs


# ---------------------------------------------------------------------------
# Altitude and descent detection
# ---------------------------------------------------------------------------

def choose_xkf_position_frame(data: Dict[str, pd.DataFrame]) -> Tuple[str, pd.DataFrame]:
    for name in ("XKF1", "XKF0", "XKF2"):
        df = data.get(name, pd.DataFrame())
        if (
            df is not None
            and not df.empty
            and all(c in df.columns for c in ("PN", "PE", "PD"))
        ):
            return name, df
    return "", pd.DataFrame()


def make_detection_grid(
    data: Dict[str, pd.DataFrame],
    dt: float,
) -> Tuple[pd.DataFrame, str]:
    """
    Altitude preference:
      1) POS.RelHomeAlt
      2) -XKF*.PD
      3) GPS.Alt relative to a low percentile of GPS altitude

    The third option is a pragmatic fallback for landing datasets.
    """
    pos = data.get("POS", pd.DataFrame())
    if not pos.empty and "RelHomeAlt" in pos.columns:
        base = pos[["t", "RelHomeAlt"]].dropna().copy()
        source = "POS.RelHomeAlt"
        base["altitude_rel_m"] = safe_numeric(base["RelHomeAlt"])
        base = base[["t", "altitude_rel_m"]]

    else:
        xkf_name, xkf = choose_xkf_position_frame(data)
        if not xkf.empty:
            base = xkf[["t", "PD"]].dropna().copy()
            source = f"-{xkf_name}.PD"
            base["altitude_rel_m"] = -safe_numeric(base["PD"])
            base = base[["t", "altitude_rel_m"]]
        else:
            gps = data.get("GPS", pd.DataFrame())
            if gps.empty or "Alt" not in gps.columns:
                raise ValueError(
                    "Could not build relative altitude. Need POS.RelHomeAlt, "
                    "XKF*.PD, or GPS.Alt."
                )

            gps_alt = safe_numeric(gps["Alt"])
            finite = gps_alt[np.isfinite(gps_alt)]
            if finite.empty:
                raise ValueError("GPS.Alt exists but contains no usable values.")

            # A robust runway/ground proxy for logs that contain the aircraft
            # near ground at least once.
            ground_proxy_m = float(np.nanpercentile(finite.to_numpy(), 2.0))
            base = gps[["t"]].copy()
            base["altitude_rel_m"] = gps_alt - ground_proxy_m
            base = base.dropna()
            source = f"GPS.Alt-minus-p02({ground_proxy_m:.2f}m)"

    t_min = float(base["t"].min())
    t_max = float(base["t"].max())
    if t_max <= t_min:
        raise ValueError("Invalid time span in altitude source.")

    times = np.arange(t_min, t_max + dt * 0.5, dt, dtype=float)
    grid = pd.DataFrame({"t": times})
    alt = align_to_grid(base, grid, tolerance_s=max(0.25, 3.0 * dt))
    alt["altitude_rel_m"] = pd.to_numeric(
        alt["altitude_rel_m"], errors="coerce"
    ).interpolate(limit_direction="both")

    return alt[["t", "altitude_rel_m"]], source


def detect_descents(
    grid: pd.DataFrame,
    cfg: ExtractionConfig,
) -> List[Tuple[int, int]]:
    t = grid["t"].to_numpy(dtype=float)
    alt = grid["altitude_rel_m"].to_numpy(dtype=float)

    if len(t) < 3:
        return []

    smooth_n = max(1, int(round(cfg.smooth_altitude_s / cfg.dt)))
    alt_smooth = rolling_smooth(alt, smooth_n)

    # Positive means descending.
    descent_rate = -np.gradient(alt_smooth, t)

    finite = np.isfinite(alt_smooth) & np.isfinite(descent_rate)
    descending = (
        finite
        & (alt_smooth <= cfg.max_altitude_m)
        & (alt_smooth >= cfg.min_altitude_m)
        & (descent_rate >= cfg.min_descent_rate_mps)
    )

    gap_n = max(0, int(round(cfg.merge_gap_s / cfg.dt)))
    descending = fill_short_false_gaps(descending, gap_n)

    candidates = contiguous_true_regions(descending)
    valid: List[Tuple[int, int]] = []

    pad_before_n = int(round(cfg.pad_before_s / cfg.dt))
    pad_after_n = int(round(cfg.pad_after_s / cfg.dt))

    for start, end in candidates:
        duration = t[end] - t[start]
        if duration < cfg.min_duration_s:
            continue

        segment_alt = alt_smooth[start : end + 1]
        if len(segment_alt) == 0 or not np.isfinite(segment_alt).any():
            continue

        start_alt = float(alt_smooth[start])
        min_alt = float(np.nanmin(segment_alt))
        altitude_drop = start_alt - min_alt

        if altitude_drop < cfg.min_altitude_drop_m:
            continue

        s = max(0, start - pad_before_n)
        e = min(len(grid) - 1, end + pad_after_n)

        # Keep the padded window within the low-altitude approach neighborhood.
        while s < start and alt_smooth[s] > cfg.max_altitude_m * 1.10:
            s += 1

        valid.append((s, e))

    # Merge overlapping padded segments.
    merged: List[Tuple[int, int]] = []
    for s, e in valid:
        if not merged or s > merged[-1][1] + 1:
            merged.append((s, e))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))

    if cfg.final_only and merged:
        return [merged[-1]]

    return merged


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def rotation_ned_to_body(
    roll_rad: np.ndarray,
    pitch_rad: np.ndarray,
    yaw_rad: np.ndarray,
    vn: np.ndarray,
    ve: np.ndarray,
    vd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized NED -> body FRD rotation.
    """
    cphi = np.cos(roll_rad)
    sphi = np.sin(roll_rad)
    cth = np.cos(pitch_rad)
    sth = np.sin(pitch_rad)
    cpsi = np.cos(yaw_rad)
    spsi = np.sin(yaw_rad)

    u = cth * cpsi * vn + cth * spsi * ve - sth * vd
    v = (
        (sphi * sth * cpsi - cphi * spsi) * vn
        + (sphi * sth * spsi + cphi * cpsi) * ve
        + sphi * cth * vd
    )
    w = (
        (cphi * sth * cpsi + sphi * spsi) * vn
        + (cphi * sth * spsi - sphi * cpsi) * ve
        + cphi * cth * vd
    )
    return u, v, w


def channel_value(
    df: pd.DataFrame,
    prefix: str,
    channel: int,
) -> np.ndarray:
    return get_col(df, [f"{prefix}_C{channel}"])


def build_descent_dataframe(
    data: Dict[str, pd.DataFrame],
    detection_grid: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    label: str,
    log_path: Path,
    descent_index: int,
    altitude_source: str,
    cfg: ExtractionConfig,
) -> pd.DataFrame:
    segment_grid = detection_grid.iloc[start_idx : end_idx + 1][["t"]].copy()
    segment_grid = segment_grid.reset_index(drop=True)

    # Preserve raw fields from useful messages. This gives you flexibility later
    # without reparsing the BIN files.
    tolerances = {
        "GPS": 0.30,
        "POS": 0.20,
        "ATT": 0.08,
        "IMU": 0.08,
        "AHR2": 0.15,
        "XKF0": 0.10,
        "XKF1": 0.10,
        "XKF2": 0.10,
        "CTUN": 0.15,
        "NTUN": 0.25,
        "TECS": 0.25,
        "ARSP": 0.20,
        "AOA": 0.20,
        "DCM": 0.25,
        "RCIN": 0.10,
        "RCOU": 0.10,
        "BARO": 0.20,
        "RFND": 0.20,
        "MODE": 1.00,
    }

    out = segment_grid.copy()

    # Bring exact detection altitude into the cached data.
    det_segment = detection_grid.iloc[start_idx : end_idx + 1].reset_index(drop=True)
    out["altitude_rel_m"] = det_segment["altitude_rel_m"].to_numpy(dtype=float)

    for msg in MESSAGE_TYPES:
        df = data.get(msg, pd.DataFrame())
        if df is None or df.empty:
            continue
        aligned = prefixed_aligned(
            df,
            segment_grid,
            prefix=msg.lower(),
            tolerance_s=tolerances.get(msg, 0.20),
        )
        # 't' is already present in out.
        new_cols = [c for c in aligned.columns if c != "t"]
        out = pd.concat(
            [out.reset_index(drop=True), aligned[new_cols].reset_index(drop=True)],
            axis=1,
        )

    # ------------------------------------------------------------------
    # Core metadata
    # ------------------------------------------------------------------
    out.insert(0, "label", label)
    out.insert(1, "log_file", log_path.name)
    out.insert(2, "log_path", str(log_path))
    out.insert(3, "descent_id", f"{log_path.stem}__descent_{descent_index:03d}")
    out.insert(4, "descent_index", descent_index)
    out["altitude_source"] = altitude_source

    t0 = float(out["t"].iloc[0])
    out["time_from_descent_start_s"] = out["t"] - t0

    # Keep a convenient default altitude bin without throwing away raw altitude.
    out["altitude_bin_5m"] = (
        np.floor(out["altitude_rel_m"] / 5.0) * 5.0
    )

    # ------------------------------------------------------------------
    # Attitude
    # ------------------------------------------------------------------
    roll_deg = get_col(out, ["att_Roll", "ahr2_Roll", "ctun_Roll"])
    pitch_deg = get_col(out, ["att_Pitch", "ahr2_Pitch", "ctun_Pitch"])
    yaw_deg = get_col(out, ["att_Yaw", "ahr2_Yaw"])

    out["roll_deg"] = roll_deg
    out["pitch_deg"] = pitch_deg
    out["yaw_deg"] = yaw_deg

    out["desired_roll_deg"] = get_col(
        out, ["att_DesRoll", "ctun_NavRoll", "att_RollDes", "att_RDes"]
    )
    out["desired_pitch_deg"] = get_col(
        out, ["att_DesPitch", "ctun_NavPitch", "att_PitchDes", "att_PDes"]
    )
    out["desired_yaw_deg"] = get_col(
        out, ["att_DesYaw", "att_YawDes", "att_YDes", "ntun_NavBrg"]
    )

    # ------------------------------------------------------------------
    # Position / ground velocity
    # ------------------------------------------------------------------
    xkf_name, _ = choose_xkf_position_frame(data)
    xprefix = xkf_name.lower() if xkf_name else ""

    pn = get_col(out, [f"{xprefix}_PN"] if xprefix else [])
    pe = get_col(out, [f"{xprefix}_PE"] if xprefix else [])
    pd_ = get_col(out, [f"{xprefix}_PD"] if xprefix else [])

    out["pn_m"] = pn
    out["pe_m"] = pe
    out["pd_m"] = pd_

    out["lat_deg"] = get_col(out, ["pos_Lat", "gps_Lat"])
    out["lon_deg"] = get_col(out, ["pos_Lng", "gps_Lng"])

    vn = get_col(out, [f"{xprefix}_VN"] if xprefix else [])
    ve = get_col(out, [f"{xprefix}_VE"] if xprefix else [])
    vd = get_col(out, [f"{xprefix}_VD"] if xprefix else [])

    gps_spd = get_col(out, ["gps_Spd"])
    gps_course_deg = get_col(out, ["gps_GCrs", "gps_Crs"])
    gps_vz = get_col(out, ["gps_VZ"])

    # Fill missing EKF horizontal velocity from GPS speed/course.
    gps_course_rad = np.deg2rad(gps_course_deg)
    vn_gps = gps_spd * np.cos(gps_course_rad)
    ve_gps = gps_spd * np.sin(gps_course_rad)

    vn = np.where(np.isfinite(vn), vn, vn_gps)
    ve = np.where(np.isfinite(ve), ve, ve_gps)
    # GPS VZ is retained raw, but altitude derivative below is the canonical
    # descent-rate feature to avoid relying on sign convention.
    vd = np.where(np.isfinite(vd), vd, gps_vz)

    out["vn_mps"] = vn
    out["ve_mps"] = ve
    out["vd_mps_raw"] = vd
    out["groundspeed_mps"] = np.sqrt(vn**2 + ve**2)

    # ------------------------------------------------------------------
    # Descent rate: positive = descending
    # ------------------------------------------------------------------
    alt_smooth = rolling_smooth(
        out["altitude_rel_m"].to_numpy(dtype=float),
        max(1, int(round(cfg.smooth_altitude_s / cfg.dt))),
    )
    t = out["t"].to_numpy(dtype=float)
    out["altitude_rel_smooth_m"] = alt_smooth
    out["descent_rate_mps"] = -np.gradient(alt_smooth, t)

    # ------------------------------------------------------------------
    # Measured / estimated airspeed
    # ------------------------------------------------------------------
    arsp = get_col(out, ["arsp_Airspeed"])
    ctun_as = get_col(out, ["ctun_As"])
    airspeed = np.where(np.isfinite(arsp), arsp, ctun_as)
    airspeed = np.where(np.isfinite(airspeed), airspeed, gps_spd)

    out["airspeed_mps"] = airspeed
    source = np.full(len(out), "GPS_groundspeed_fallback", dtype=object)
    source[np.isfinite(ctun_as)] = "CTUN.As"
    source[np.isfinite(arsp)] = "ARSP.Airspeed"
    out["airspeed_source"] = source

    # ------------------------------------------------------------------
    # Body-frame ground / air-relative velocity and alpha/beta
    # ------------------------------------------------------------------
    roll_rad = np.deg2rad(roll_deg)
    pitch_rad = np.deg2rad(pitch_deg)
    yaw_rad = wrap_pi(np.deg2rad(yaw_deg))

    u_g, v_g, w_g = rotation_ned_to_body(
        roll_rad, pitch_rad, yaw_rad, vn, ve, vd
    )
    out["u_ground_body_mps"] = u_g
    out["v_ground_body_mps"] = v_g
    out["w_ground_body_mps"] = w_g

    wind_n = get_col(out, ["dcm_VWN"])
    wind_e = get_col(out, ["dcm_VWE"])
    wind_d = get_col(out, ["dcm_VWD"])

    have_wind = (
        np.isfinite(wind_n)
        & np.isfinite(wind_e)
        & np.isfinite(wind_d)
    )

    air_vn = np.where(have_wind, vn - wind_n, vn)
    air_ve = np.where(have_wind, ve - wind_e, ve)
    air_vd = np.where(have_wind, vd - wind_d, vd)

    u_a, v_a, w_a = rotation_ned_to_body(
        roll_rad, pitch_rad, yaw_rad, air_vn, air_ve, air_vd
    )

    out["wind_n_mps"] = wind_n
    out["wind_e_mps"] = wind_e
    out["wind_d_mps"] = wind_d
    out["u_air_body_mps"] = u_a
    out["v_air_body_mps"] = v_a
    out["w_air_body_mps"] = w_a

    speed_vec = np.sqrt(u_a**2 + v_a**2 + w_a**2)

    with np.errstate(invalid="ignore", divide="ignore"):
        alpha_rad = np.arctan2(w_a, u_a)
        beta_arg = np.divide(
            v_a,
            speed_vec,
            out=np.full_like(v_a, np.nan),
            where=speed_vec > 1e-6,
        )
        beta_rad = np.arcsin(np.clip(beta_arg, -1.0, 1.0))

    # Preserve reconstructed aerodynamic angles for diagnostics / fallback.
    aoa_derived_deg = np.rad2deg(alpha_rad)
    sideslip_derived_deg = np.rad2deg(beta_rad)

    out["aoa_derived_deg"] = aoa_derived_deg
    out["sideslip_derived_deg"] = sideslip_derived_deg

    derived_source = np.where(
        have_wind,
        "wind_corrected_DCM",
        "ground_relative_approx",
    )
    out["aero_angle_derived_source"] = derived_source

    # ------------------------------------------------------------------
    # Primary AoA / sideslip: use ArduPilot's logged AOA message directly.
    #
    # After prefixing the AOA DataFlash message:
    #   AOA.AOA -> aoa_AOA
    #   AOA.SSA -> aoa_SSA
    #
    # If either value is unavailable for a row, fall back independently to
    # the reconstructed value above.
    # ------------------------------------------------------------------
    aoa_logged_deg = get_col(out, ["aoa_AOA"])
    sideslip_logged_deg = get_col(out, ["aoa_SSA"])

    out["aoa_logged_deg"] = aoa_logged_deg
    out["sideslip_logged_deg"] = sideslip_logged_deg

    have_logged_aoa = np.isfinite(aoa_logged_deg)
    have_logged_ssa = np.isfinite(sideslip_logged_deg)

    out["aoa_deg"] = np.where(
        have_logged_aoa,
        aoa_logged_deg,
        aoa_derived_deg,
    )
    out["sideslip_deg"] = np.where(
        have_logged_ssa,
        sideslip_logged_deg,
        sideslip_derived_deg,
    )

    out["aoa_source"] = np.where(
        have_logged_aoa,
        "ArduPilot_AOA.AOA",
        derived_source,
    )
    out["sideslip_source"] = np.where(
        have_logged_ssa,
        "ArduPilot_AOA.SSA",
        derived_source,
    )

    # Backward-compatible combined source field.
    # Most rows should have both AOA and SSA logged together.
    out["aero_angle_source"] = np.where(
        have_logged_aoa & have_logged_ssa,
        "ArduPilot_AOA",
        np.where(
            (~have_logged_aoa) & (~have_logged_ssa),
            derived_source,
            "mixed_logged_and_derived",
        ),
    )

    # ------------------------------------------------------------------
    # IMU rates / accelerations
    # ------------------------------------------------------------------
    out["p_radps"] = get_col(out, ["imu_GyrX", "imu_GX"])
    out["q_radps"] = get_col(out, ["imu_GyrY", "imu_GY"])
    out["r_radps"] = get_col(out, ["imu_GyrZ", "imu_GZ"])
    out["accel_x_mps2"] = get_col(out, ["imu_AccX", "imu_AX"])
    out["accel_y_mps2"] = get_col(out, ["imu_AccY", "imu_AY"])
    out["accel_z_mps2"] = get_col(out, ["imu_AccZ", "imu_AZ"])

    # ------------------------------------------------------------------
    # Pilot inputs (RCIN) and actuator outputs (RCOU)
    # ------------------------------------------------------------------
    roll_pwm = channel_value(out, "rcin", cfg.roll_channel)
    pitch_pwm = channel_value(out, "rcin", cfg.pitch_channel)
    throttle_pwm = channel_value(out, "rcin", cfg.throttle_channel)
    yaw_pwm = channel_value(out, "rcin", cfg.yaw_channel)

    out["pilot_roll_pwm"] = roll_pwm
    out["pilot_pitch_pwm"] = pitch_pwm
    out["pilot_throttle_pwm"] = throttle_pwm
    out["pilot_yaw_pwm"] = yaw_pwm

    out["pilot_roll_centered"] = (roll_pwm - 1500.0) / 500.0
    out["pilot_pitch_centered"] = (pitch_pwm - 1500.0) / 500.0
    out["pilot_yaw_centered"] = (yaw_pwm - 1500.0) / 500.0
    out["pilot_throttle_norm"] = (throttle_pwm - 1000.0) / 1000.0

    out["servo_roll_pwm"] = channel_value(out, "rcou", cfg.roll_channel)
    out["servo_pitch_pwm"] = channel_value(out, "rcou", cfg.pitch_channel)
    out["servo_throttle_pwm"] = channel_value(out, "rcou", cfg.throttle_channel)
    out["servo_yaw_pwm"] = channel_value(out, "rcou", cfg.yaw_channel)

    # Useful command/output features from Plane CTUN when present.
    out["throttle_output"] = get_col(out, ["ctun_ThO"])
    out["rudder_output"] = get_col(out, ["ctun_RdO"])
    out["tecs_throttle_demand"] = get_col(out, ["ctun_ThD"])

    # ------------------------------------------------------------------
    # Correction/activity features
    # ------------------------------------------------------------------
    def derivative(values: np.ndarray) -> np.ndarray:
        values = values.astype(float)
        if len(values) < 2:
            return np.full_like(values, np.nan)
        return np.gradient(values, t)

    out["pilot_roll_rate_per_s"] = derivative(out["pilot_roll_centered"].to_numpy())
    out["pilot_pitch_rate_per_s"] = derivative(out["pilot_pitch_centered"].to_numpy())
    out["pilot_yaw_rate_per_s"] = derivative(out["pilot_yaw_centered"].to_numpy())
    out["pilot_throttle_rate_per_s"] = derivative(out["pilot_throttle_norm"].to_numpy())

    out["roll_error_deg"] = out["desired_roll_deg"] - out["roll_deg"]
    out["pitch_error_deg"] = out["desired_pitch_deg"] - out["pitch_deg"]

    # Sort columns with the analysis-critical fields first.
    priority = [
        "label",
        "log_file",
        "log_path",
        "descent_id",
        "descent_index",
        "t",
        "time_from_descent_start_s",
        "altitude_source",
        "altitude_rel_m",
        "altitude_rel_smooth_m",
        "altitude_bin_5m",
        "descent_rate_mps",
        "airspeed_mps",
        "airspeed_source",
        "aoa_deg",
        "aoa_logged_deg",
        "aoa_derived_deg",
        "aoa_source",
        "sideslip_deg",
        "sideslip_logged_deg",
        "sideslip_derived_deg",
        "sideslip_source",
        "aero_angle_source",
        "aero_angle_derived_source",
        "groundspeed_mps",
        "vn_mps",
        "ve_mps",
        "vd_mps_raw",
        "roll_deg",
        "pitch_deg",
        "yaw_deg",
        "desired_roll_deg",
        "desired_pitch_deg",
        "desired_yaw_deg",
        "roll_error_deg",
        "pitch_error_deg",
        "pilot_roll_pwm",
        "pilot_pitch_pwm",
        "pilot_throttle_pwm",
        "pilot_yaw_pwm",
        "pilot_roll_centered",
        "pilot_pitch_centered",
        "pilot_throttle_norm",
        "pilot_yaw_centered",
        "pilot_roll_rate_per_s",
        "pilot_pitch_rate_per_s",
        "pilot_throttle_rate_per_s",
        "pilot_yaw_rate_per_s",
        "servo_roll_pwm",
        "servo_pitch_pwm",
        "servo_throttle_pwm",
        "servo_yaw_pwm",
        "throttle_output",
        "rudder_output",
        "tecs_throttle_demand",
        "p_radps",
        "q_radps",
        "r_radps",
        "accel_x_mps2",
        "accel_y_mps2",
        "accel_z_mps2",
        "lat_deg",
        "lon_deg",
        "pn_m",
        "pe_m",
        "pd_m",
        "wind_n_mps",
        "wind_e_mps",
        "wind_d_mps",
        "u_air_body_mps",
        "v_air_body_mps",
        "w_air_body_mps",
        "u_ground_body_mps",
        "v_ground_body_mps",
        "w_ground_body_mps",
    ]

    remaining = [c for c in out.columns if c not in priority]
    return out[[c for c in priority if c in out.columns] + remaining]


# ---------------------------------------------------------------------------
# Cache writing and manifest
# ---------------------------------------------------------------------------

def write_cache(df: pd.DataFrame, base_path: Path) -> Path:
    base_path.parent.mkdir(parents=True, exist_ok=True)

    parquet_path = base_path.with_suffix(".parquet")
    try:
        df.to_parquet(parquet_path, index=False)
        return parquet_path
    except (ImportError, ModuleNotFoundError):
        csv_path = base_path.with_suffix(".csv.gz")
        df.to_csv(csv_path, index=False, compression="gzip")
        return csv_path


def make_manifest_record(
    df: pd.DataFrame,
    cache_path: Path,
    log_path: Path,
    label: str,
    descent_index: int,
    altitude_source: str,
) -> Dict[str, Any]:
    return {
        "label": label,
        "log_file": log_path.name,
        "log_path": str(log_path),
        "descent_index": descent_index,
        "descent_id": str(df["descent_id"].iloc[0]),
        "cache_path": str(cache_path),
        "samples": int(len(df)),
        "start_t_s": float(df["t"].iloc[0]),
        "end_t_s": float(df["t"].iloc[-1]),
        "duration_s": float(df["t"].iloc[-1] - df["t"].iloc[0]),
        "start_altitude_m": float(df["altitude_rel_m"].iloc[0]),
        "minimum_altitude_m": float(np.nanmin(df["altitude_rel_m"])),
        "altitude_drop_m": float(
            df["altitude_rel_m"].iloc[0] - np.nanmin(df["altitude_rel_m"])
        ),
        "max_descent_rate_mps": float(np.nanmax(df["descent_rate_mps"])),
        "mean_airspeed_mps": float(np.nanmean(df["airspeed_mps"])),
        "altitude_source": altitude_source,
        "aoa_logged_fraction": float(
            np.mean(df["aoa_source"] == "ArduPilot_AOA.AOA")
        ),
        "sideslip_logged_fraction": float(
            np.mean(df["sideslip_source"] == "ArduPilot_AOA.SSA")
        ),
        "aoa_derived_wind_corrected_fraction": float(
            np.mean(df["aero_angle_derived_source"] == "wind_corrected_DCM")
        ),
    }


def process_log(
    log_path: Path,
    label: str,
    output_root: Path,
    cfg: ExtractionConfig,
    verbose: bool = False,
) -> List[Dict[str, Any]]:
    parser = FlightParser(log_path, verbose=verbose)
    data = parser.read_messages(MESSAGE_TYPES)

    detection_grid, altitude_source = make_detection_grid(data, cfg.dt)
    descents = detect_descents(detection_grid, cfg)

    if verbose:
        print(
            f"[{label}] {log_path.name}: "
            f"{len(descents)} descent segment(s), altitude={altitude_source}"
        )

    records: List[Dict[str, Any]] = []

    for descent_index, (start_idx, end_idx) in enumerate(descents):
        df = build_descent_dataframe(
            data=data,
            detection_grid=detection_grid,
            start_idx=start_idx,
            end_idx=end_idx,
            label=label,
            log_path=log_path,
            descent_index=descent_index,
            altitude_source=altitude_source,
            cfg=cfg,
        )

        base = output_root / label / f"{log_path.stem}__descent_{descent_index:03d}"
        cache_path = write_cache(df, base)

        records.append(
            make_manifest_record(
                df=df,
                cache_path=cache_path,
                log_path=log_path,
                label=label,
                descent_index=descent_index,
                altitude_source=altitude_source,
            )
        )

    return records


# ---------------------------------------------------------------------------
# Dataset crawl
# ---------------------------------------------------------------------------

def find_bin_files(root: Path, label: str) -> List[Path]:
    folder = root / label
    if not folder.exists():
        return []

    files = list(folder.rglob("*.BIN")) + list(folder.rglob("*.bin"))
    # Avoid duplicates on case-insensitive filesystems.
    return sorted(set(p.resolve() for p in files))


def run_dataset(
    input_root: Path,
    output_root: Path,
    labels: Sequence[str],
    cfg: ExtractionConfig,
    verbose: bool = False,
) -> pd.DataFrame:
    output_root.mkdir(parents=True, exist_ok=True)

    all_manifest: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for label in labels:
        logs = find_bin_files(input_root, label)
        print(f"{label}: found {len(logs)} BIN file(s)")

        for i, log_path in enumerate(logs, start=1):
            print(f"  [{i}/{len(logs)}] {log_path.name}")

            try:
                records = process_log(
                    log_path=log_path,
                    label=label,
                    output_root=output_root,
                    cfg=cfg,
                    verbose=verbose,
                )
                all_manifest.extend(records)

                if not records:
                    print("      no qualifying descent found")

            except Exception as exc:
                print(f"      ERROR: {exc}")
                errors.append(
                    {
                        "label": label,
                        "log_path": str(log_path),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )

    manifest = pd.DataFrame(all_manifest)
    manifest_path = output_root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    config_path = output_root / "extraction_config.json"
    config_path.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    if errors:
        errors_path = output_root / "errors.json"
        errors_path.write_text(json.dumps(errors, indent=2), encoding="utf-8")
        print(f"\nCompleted with {len(errors)} error(s): {errors_path}")

    print(f"\nCached {len(manifest)} descent segment(s)")
    print(f"Manifest: {manifest_path}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract and cache landing descent segments from ArduPilot BIN logs."
    )

    p.add_argument(
        "--input",
        type=Path,
        default=Path("data/landings"),
        help="Root containing expert/ and fail/ (default: data/landings)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("data/landing_cache"),
        help="Cache output directory (default: data/landing_cache)",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=["expert", "fail"],
        help="Class directories to crawl (default: expert fail)",
    )

    p.add_argument("--dt", type=float, default=0.05, help="Cache sample period in seconds")
    p.add_argument(
        "--max-altitude",
        type=float,
        default=120.0,
        help="Only detect descent below this relative altitude in meters",
    )
    p.add_argument(
        "--min-altitude",
        type=float,
        default=-10.0,
        help="Lowest allowed relative altitude during detection",
    )
    p.add_argument(
        "--min-descent-rate",
        type=float,
        default=0.25,
        help="Positive-down descent-rate threshold in m/s",
    )
    p.add_argument(
        "--min-duration",
        type=float,
        default=5.0,
        help="Minimum qualifying descent duration in seconds",
    )
    p.add_argument(
        "--min-drop",
        type=float,
        default=8.0,
        help="Minimum altitude loss for a qualifying descent in meters",
    )
    p.add_argument(
        "--merge-gap",
        type=float,
        default=5.0,
        help="Merge temporary non-descending gaps up to this many seconds",
    )
    p.add_argument(
        "--pad-before",
        type=float,
        default=2.0,
        help="Seconds to retain before detected descent",
    )
    p.add_argument(
        "--pad-after",
        type=float,
        default=5.0,
        help="Seconds to retain after detected descent to capture flare/touchdown",
    )
    p.add_argument(
        "--final-only",
        action="store_true",
        help="Keep only the last detected low-altitude descent from each BIN",
    )

    p.add_argument("--roll-channel", type=int, default=DEFAULT_ROLL_CH)
    p.add_argument("--pitch-channel", type=int, default=DEFAULT_PITCH_CH)
    p.add_argument("--throttle-channel", type=int, default=DEFAULT_THROTTLE_CH)
    p.add_argument("--yaw-channel", type=int, default=DEFAULT_YAW_CH)

    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = ExtractionConfig(
        dt=args.dt,
        max_altitude_m=args.max_altitude,
        min_altitude_m=args.min_altitude,
        min_descent_rate_mps=args.min_descent_rate,
        min_duration_s=args.min_duration,
        min_altitude_drop_m=args.min_drop,
        merge_gap_s=args.merge_gap,
        pad_before_s=args.pad_before,
        pad_after_s=args.pad_after,
        final_only=args.final_only,
        roll_channel=args.roll_channel,
        pitch_channel=args.pitch_channel,
        throttle_channel=args.throttle_channel,
        yaw_channel=args.yaw_channel,
    )

    run_dataset(
        input_root=args.input,
        output_root=args.output,
        labels=args.labels,
        cfg=cfg,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
