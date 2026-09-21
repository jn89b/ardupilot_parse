# Capturing Landing Trends

## Goal
Compare good vs. bad landings to identify what precedes a poor outcome — specifically whether operator stick corrections appear before instability, and whether "bad landing" signatures differ from outright crash signatures.

## Dataset Classes
Source logs live under `data/landings/`:
- `expert/` — clean, successful landings (baseline/"good" class).
- `fail/` — off-nominal landings, including crash events.

Within `fail/`, distinguish two sub-cases when labeling:
- `bad` — landing completed but with corrective/off-nominal behavior.
- `crash` — landing did not complete safely (aircraft damage / hard impact).

A manifest mapping each log file to a label (`good` / `bad` / `crash`) should be maintained before running feature extraction, since all downstream comparisons key off this label.

## Touchdown Window
Each log must be aligned to a common reference point — touchdown — rather than compared over the whole flight:
- Reference (t=0): estimated touchdown time, detected from altitude (EKF `PD`, i.e. `pd_m`) crossing near ground level and/or descent rate flattening to ~0.
- Window: approximately 5–10 s before touchdown through 2–5 s after.
- All variables are resampled onto this shared, touchdown-relative time axis so events across logs are directly comparable.

## Variables of Interest
Extracted per-timestep via `FlightParser.get_desired_data` / `export_json_timeseries`:
- Altitude / descent rate — from EKF `PD` (`pd_m`) and its derivative.
- Airspeed — from `NTUN` or GPS speed.
- Angle of attack — from `NTUN`.
- Sideslip — from `NTUN`.
- Attitude (roll/pitch/yaw) — from `ATT` / `AHR2`.
- Commanded attitude — desired attitude fields used to detect operator stick corrections.
- Throttle command — to detect go-around or power corrections.
- Body rates — roll/pitch/yaw rates to detect oscillation near touchdown.

## Key Questions
1. Does a corrective or larger-than-normal stick command appear before the aircraft state becomes unstable?
2. Which variables diverge earliest between `expert` and `fail` cases as altitude approaches zero?
3. Is there a measurable difference between the `bad`-but-recovered landings and the `crash` case?

## Workflow
1. Label the dataset and confirm the classes under `data/landings/`.
2. Export each `.BIN` log to a uniform timeseries using `generate_json.py`.
3. Align each series to touchdown and trim to the landing window.
4. Extract per-landing features: min/max altitude, peak descent rate, mean airspeed pre-flare, peak AOA, sideslip variance, and timing/magnitude of commanded-attitude changes.
5. Compare classes by building a combined table and grouped plots.
6. Validate with a few raw-log spot checks before drawing conclusions.