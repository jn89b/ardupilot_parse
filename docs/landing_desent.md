# Landing Descent Cache

This script crawls ArduPilot `.BIN` logs under:

```text
data/landings/
├── expert/
│   ├── flight_001.BIN
│   └── ...
└── fail/
    ├── flight_101.BIN
    └── ...
```

It detects landing descent segments and caches analysis-ready data including:

* Altitude
* Descent rate
* Airspeed
* Groundspeed
* Angle of attack
* Sideslip
* Roll / pitch / yaw
* Desired attitude
* Pilot RC inputs
* Servo outputs
* Throttle
* IMU rates and accelerations
* EKF position and velocity

## Install Dependencies

```bash
pip install pymavlink pandas numpy pyarrow
```

If using `uv`:

```bash
uv add pymavlink pandas numpy pyarrow
```

## Run

Basic usage:

```bash
python cache_landing_descents.py
```

By default, the script reads:

```text
data/landings/
```

and writes cached data to:

```text
data/landing_cache/
```

## Recommended Usage

If each `.BIN` contains a complete flight and you only want the final landing descent:

```bash
python cache_landing_descents.py \
    --final-only \
    --max-altitude 100
```

This analyzes the final descent below `100 m` relative altitude.

## Useful Options

```bash
python cache_landing_descents.py \
    --final-only \
    --max-altitude 100 \
    --min-descent-rate 0.25 \
    --min-duration 5 \
    --min-drop 8 \
    --merge-gap 5 \
    --pad-before 2 \
    --pad-after 5
```

| Argument             | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `--max-altitude`     | Maximum relative altitude to consider part of the landing |
| `--min-descent-rate` | Minimum descent rate in m/s                               |
| `--min-duration`     | Minimum descent duration in seconds                       |
| `--min-drop`         | Minimum total altitude loss                               |
| `--merge-gap`        | Merge short interruptions in the descent                  |
| `--pad-before`       | Keep additional seconds before detected descent           |
| `--pad-after`        | Keep additional seconds after descent                     |
| `--final-only`       | Only cache the final descent from each flight             |

## Output

Example:

```text
data/landing_cache/
├── expert/
│   ├── flight_001__descent_000.parquet
│   └── flight_002__descent_000.parquet
├── fail/
│   ├── flight_101__descent_000.parquet
│   └── flight_102__descent_000.parquet
├── manifest.csv
└── extraction_config.json
```

Each descent is stored independently so later analysis can compare **expert vs. failed landings as a function of altitude**.

`manifest.csv` provides a summary of every detected descent.

## Load Cached Data

```python
import pandas as pd

df = pd.read_parquet(
    "data/landing_cache/expert/flight_001__descent_000.parquet"
)

print(df.head())
```

The cached files can then be used for altitude-binned distributions of airspeed, AoA, sideslip, descent rate, and pilot control behavior without reparsing the original `.BIN` logs.
