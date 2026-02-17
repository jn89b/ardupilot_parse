import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Dict, Any, Tuple

import pandas as pd
from pymavlink import DFReader

# -------------------------
# Alignment utilities
# -------------------------
def align_to_grid(df: pd.DataFrame, t_grid: pd.DataFrame, tol_s: float) -> pd.DataFrame:
    """
    Align df to a uniform time grid using nearest-neighbor with tolerance.

    If no sample is within tol_s, fields will be NaN (prevents "frozen" values).
    """
    if df is None or df.empty or "t" not in df:
        return t_grid.copy()

    df2 = df.sort_values("t").copy()
    out = pd.merge_asof(t_grid, df2, on="t", direction="nearest", tolerance=tol_s)
    return out


def pick_nkf_with_cols(desired_data: Dict[str, pd.DataFrame], cols=("PN", "PE", "PD")) -> pd.DataFrame:
    """
    Find which NKF message contains PN/PE/PD for this log.
    """
    for name in ["NKF1", "NKF2", "NKF3"]:
        df = desired_data.get(name, pd.DataFrame())
        if not df.empty and all(c in df.columns for c in cols):
            return df
    return pd.DataFrame()


# -------------------------
# Core parser
# -------------------------
class FlightParser:
    def __init__(self, log_path: str, verbose: bool = False) -> None:
        self.log_path: str = log_path
        self.binary_log = DFReader.DFReader_binary(filename=log_path)

        if verbose:
            print("Message types present:", sorted([fmt.name for fmt in self.binary_log.formats.values()]))
            print(
                "Counts by type (non-zero):",
                {
                    self.binary_log.id_to_name[i]: c
                    for i, c in enumerate(self.binary_log.counts)
                    if c > 0 and i in self.binary_log.id_to_name
                },
            )

        self.binary_log.rewind()

    @staticmethod
    def _safe_get(row: pd.Series, candidates: List[str], default: float = 0.0) -> float:
        for c in candidates:
            if c in row and pd.notna(row[c]):
                try:
                    return float(row[c])
                except Exception:
                    pass
        return float(default)

    @staticmethod
    def _deg2rad(x_deg: float) -> float:
        return float(x_deg) * math.pi / 180.0

    @staticmethod
    def _wrap_pi(rad: float) -> float:
        return (rad + math.pi) % (2 * math.pi) - math.pi

    def get_desired_data(self, types: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Extract message types from the BIN into DataFrames.
        IMPORTANT: Use ONE global t0 across ALL types so times align correctly.
        """
        if not types:
            raise ValueError("Types list cannot be empty.")

        rows: Dict[str, List[Dict[str, Any]]] = {t: [] for t in types}

        while True:
            m = self.binary_log.recv_msg()
            if m is None:
                break
            mt = m.get_type()
            if mt in types:
                d = m.to_dict()
                d["TimeUS"] = getattr(m, "TimeUS", None)
                d["_t"] = getattr(m, "_timestamp", None)
                rows[mt].append(d)

        dfs = {t: pd.DataFrame(rows[t]) for t in types}

        # Global t0 from earliest TimeUS across all extracted types
        timeus_series = []
        for df in dfs.values():
            if not df.empty and "TimeUS" in df.columns:
                timeus_series.append(df["TimeUS"].dropna())

        if not timeus_series:
            raise ValueError("No TimeUS found in any selected message types.")

        t0_us = float(pd.concat(timeus_series).min())

        # Add aligned relative time to each DF
        for k, df in dfs.items():
            if df is None or df.empty or "TimeUS" not in df.columns:
                continue
            df2 = df.sort_values("TimeUS").copy()
            df2["t"] = (df2["TimeUS"] - t0_us) / 1e6
            dfs[k] = df2

        return dfs

    def export_json_timeseries(
        self,
        desired_data: Dict[str, pd.DataFrame],
        out_path: str,
        dt: float = 0.05,
        target_xyz: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        strict_nkf: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Export JSON series using:
          - Position: NKF PN/PE/PD (meters)  [North/East/Down]
          - Attitude: ATT Roll/Pitch/Yaw (deg -> rad)
          - Setpoints: ATT DesRoll/DesPitch/DesYaw if present
          - Velocities: GPS Spd + GCrs (fallback 0)
          - Rates: IMU GyrX/GyrY/GyrZ (fallback 0)

        strict_nkf:
          - True: require NKF PN/PE/PD; error if missing
          - False: allow zeros if NKF missing
        """
        gps = desired_data.get("GPS", pd.DataFrame())
        att = desired_data.get("ATT", pd.DataFrame())
        imu = desired_data.get("IMU", pd.DataFrame())
        ahr2 = desired_data.get("AHR2", pd.DataFrame())
        ctun = desired_data.get("CTUN", pd.DataFrame())

        # Prefer NKF as time base if present, otherwise fallback
        nkf = pick_nkf_with_cols(desired_data, cols=("PN", "PE", "PD"))
        base = nkf if not nkf.empty else (gps if not gps.empty else (att if not att.empty else (imu if not imu.empty else ahr2)))

        if base is None or base.empty or "t" not in base.columns:
            raise ValueError("No usable time base found (need at least one message with TimeUS -> t).")

        if strict_nkf and nkf.empty:
            raise ValueError("NKF1/NKF2/NKF3 with PN/PE/PD not found in desired_data (strict_nkf=True).")

        t_min = float(base["t"].min())
        t_max = float(base["t"].max())
        t_grid = pd.DataFrame({"t": [t_min + i * dt for i in range(int((t_max - t_min) / dt) + 1)]})

        # Align all channels to the time grid with sensible tolerances
        nkf_i = align_to_grid(nkf, t_grid, tol_s=0.05) if not nkf.empty else t_grid.copy()
        gps_i = align_to_grid(gps, t_grid, tol_s=0.2)
        att_i = align_to_grid(att, t_grid, tol_s=0.02)
        imu_i = align_to_grid(imu, t_grid, tol_s=0.02)
        ahr2_i = align_to_grid(ahr2, t_grid, tol_s=0.1)
        ctun_i = align_to_grid(ctun, t_grid, tol_s=0.1)

        tx, ty, tz = target_xyz

        series: List[Dict[str, Any]] = []
        for i in range(len(t_grid)):
            t = float(t_grid.loc[i, "t"])

            # --- Position from EKF (NED, meters) ---
            pos_north = self._safe_get(nkf_i.loc[i], ["PN"], default=0.0)
            pos_east = self._safe_get(nkf_i.loc[i], ["PE"], default=0.0)
            pos_down = self._safe_get(nkf_i.loc[i], ["PD"], default=0.0)

            # Keep legacy x/y/z if you want; here we make them consistent with NED:
            # (Many people prefer x=east, y=north, z=down; choose one and be consistent.)
            x = pos_east
            y = pos_north
            z = pos_down

            # --- Attitude (deg -> rad) ---
            roll_deg = self._safe_get(att_i.loc[i], ["Roll", "roll"], default=0.0)
            pitch_deg = self._safe_get(att_i.loc[i], ["Pitch", "pitch"], default=0.0)
            yaw_deg = self._safe_get(att_i.loc[i], ["Yaw", "yaw"], default=0.0)

            phi = self._deg2rad(roll_deg)
            theta = self._deg2rad(pitch_deg)
            psi = self._wrap_pi(self._deg2rad(yaw_deg))

            # --- Setpoints (if present) ---
            des_roll_deg = self._safe_get(att_i.loc[i], ["DesRoll", "RollDes", "RDes"], default=roll_deg)
            des_pitch_deg = self._safe_get(att_i.loc[i], ["DesPitch", "PitchDes", "PDes"], default=pitch_deg)
            des_yaw_deg = self._safe_get(att_i.loc[i], ["DesYaw", "YawDes", "YDes"], default=yaw_deg)

            phi_sp = self._deg2rad(des_roll_deg)
            theta_sp = self._deg2rad(des_pitch_deg)
            psi_sp = self._wrap_pi(self._deg2rad(des_yaw_deg))

            # --- Velocities (N/E from GPS Spd+GCrs) ---
            spd = self._safe_get(gps_i.loc[i], ["Spd", "Speed"], default=0.0)
            gcrs_deg = self._safe_get(gps_i.loc[i], ["GCrs", "Crs"], default=0.0)
            gcrs = math.radians(gcrs_deg)

            vx = float(spd * math.cos(gcrs))  # North
            vy = float(spd * math.sin(gcrs))  # East
            vz = self._safe_get(gps_i.loc[i], ["VZ", "VelZ", "Vz"], default=0.0)

            # --- Body-ish velocities (best-effort from AHR2 if present) ---
            u = self._safe_get(ahr2_i.loc[i], ["Vx", "VN", "VelN"], default=vx)
            v = self._safe_get(ahr2_i.loc[i], ["Vy", "VE", "VelE"], default=vy)
            w = self._safe_get(ahr2_i.loc[i], ["Vz", "VD", "VelD"], default=vz)

            # --- Rates p,q,r ---
            p = self._safe_get(imu_i.loc[i], ["GyrX", "p", "P"], default=0.0)
            q = self._safe_get(imu_i.loc[i], ["GyrY", "q", "Q"], default=0.0)
            r = self._safe_get(imu_i.loc[i], ["GyrZ", "r", "R"], default=0.0)

            throttle = self._safe_get(ctun_i.loc[i], ["Throttle", "ThD"], default=0.0)

            rec = {
                "timestamp": t,

                # Your existing schema
                "x": float(x),
                "y": float(y),
                "z": float(z),

                # Explicit EKF NED (requested)
                "pn_m": float(pos_north),
                "pe_m": float(pos_east),
                "pd_m": float(pos_down),

                "phi_rad": float(phi),
                "theta_rad": float(theta),
                "psi_rad": float(psi),

                "vx": float(vx),
                "vy": float(vy),
                "vz": float(vz),

                "u_ms": float(u),
                "v_ms": float(v),
                "w_ms": float(w),

                "p_radps": float(p),
                "q_radps": float(q),
                "r_radps": float(r),

                "phi_sp_rad": float(phi_sp),
                "theta_sp_rad": float(theta_sp),
                "psi_sp_rad": float(psi_sp),

                "throttle_cmd": float(throttle),
                "target_x": float(tx),
                "target_y": float(ty),
                "target_z": float(tz),
            }
            series.append(rec)

        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(series, f, indent=2)

        return series


# -------------------------
# Multiprocessing: one process per log (recommended)
# -------------------------
def process_one_log(job: Tuple[str, str, float, bool, bool]) -> str:
    log_path, out_path, dt, strict_nkf, verbose = job
    fp = FlightParser(log_path=log_path, verbose=verbose)

    desired = fp.get_desired_data(
        types=["GPS", "NTUN", "CMD", "MODE", "IMU", 
               "ATT", "AHR2", "NKF1", "NKF2", "NKF3", "CTUN"]
    )

    fp.export_json_timeseries(
        desired_data=desired,
        out_path=out_path,
        dt=dt,
        target_xyz=(0.0, 0.0, 0.0),
        strict_nkf=strict_nkf,
    )
    return out_path


def batch_export_logs(log_paths: List[str], out_dir: str, dt: float, workers: int, strict_nkf: bool, verbose: bool) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)

    jobs: List[Tuple[str, str, float, bool, bool]] = []
    for p in log_paths:
        base = os.path.splitext(os.path.basename(p))[0]
        out_path = os.path.join(out_dir, f"{base}.json")
        jobs.append((p, out_path, dt, strict_nkf, verbose))

    outs: List[str] = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(process_one_log, j) for j in jobs]
        for f in as_completed(futs):
            outs.append(f.result())

    return outs



