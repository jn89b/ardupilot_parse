# Analyze Landing Data

The `analyze_landings.py` script generates statistical tables and plots from the cached landing datasets created by `cache_landing_descents.py`.

It can analyze any landing class, such as:

```text
data/landing_cache/
├── expert/
└── fail/
```

## Install Dependencies

Using `uv`:

```bash
uv add pandas numpy matplotlib pyarrow
```

Or with `pip`:

```bash
pip install pandas numpy matplotlib pyarrow
```

## Analyze Expert Landings

```bash
uv run analyze_landings.py --label expert
```

This reads:

```text
data/landing_cache/expert/
```

and saves the results to:

```text
data/landing_analysis/expert/
```

## Analyze Failed Landings

```bash
uv run analyze_landings.py --label fail
```

This reads:

```text
data/landing_cache/fail/
```

and saves the results to:

```text
data/landing_analysis/fail/
```

## Recommended Command

To analyze both datasets over the same landing altitude range:

```bash
uv run analyze_landings.py \
    --label expert \
    --min-altitude 0 \
    --max-altitude 40 \
    --bin-width 5
```

Then run:

```bash
uv run analyze_landings.py \
    --label fail \
    --min-altitude 0 \
    --max-altitude 40 \
    --bin-width 5
```

Using the same altitude range and bin width is important when comparing the expert and failed landing distributions.

## Output

The analysis is saved under:

```text
data/landing_analysis/
├── expert/
│   ├── altitude_stats.parquet
│   ├── altitude_stats.csv
│   ├── per_descent_bin_stats.parquet
│   ├── raw_sample_altitude_stats.parquet
│   ├── overall_stats.parquet
│   ├── overall_stats.csv
│   ├── analysis_config.json
│   └── plots/
│
└── fail/
    ├── altitude_stats.parquet
    ├── altitude_stats.csv
    ├── per_descent_bin_stats.parquet
    ├── raw_sample_altitude_stats.parquet
    ├── overall_stats.parquet
    ├── overall_stats.csv
    ├── analysis_config.json
    └── plots/
```

## Important Output Files

### `altitude_stats.csv`

Primary table for analyzing behavior as a function of altitude.

It contains statistics such as:

```text
Altitude Bin
Variable
Number of Landings
Mean
Standard Deviation
Median
5th Percentile
25th Percentile
75th Percentile
95th Percentile
```

This is the main file to use when comparing expert and failed landings.

### `per_descent_bin_stats.parquet`

Contains statistics for each individual landing within each altitude bin.

This is useful for studying landing-to-landing variation.

### `raw_sample_altitude_stats.parquet`

Contains statistics calculated using all individual samples.

This is useful for inspecting the raw distributions but should generally not be the primary comparison between flights.

### `overall_stats.csv`

Provides a high-level summary of each analyzed variable across the landing dataset.

## Generated Plots

Plots are stored in:

```text
data/landing_analysis/<label>/plots/
```

Examples include:

```text
airspeed_mps_vs_altitude.png
aoa_deg_vs_altitude.png
sideslip_deg_vs_altitude.png
descent_rate_mps_vs_altitude.png

airspeed_mps_histogram.png
aoa_deg_histogram.png

airspeed_mps_landing_means_histogram.png

landing_coverage_vs_altitude.png
correlation_matrix.png
```

The altitude plots show:

```text
5th percentile
25th percentile
Mean
75th percentile
95th percentile
```

allowing an expected landing envelope to be constructed for each variable.

## Useful Options

Change the altitude-bin size:

```bash
uv run analyze_landings.py \
    --label expert \
    --bin-width 2.5
```

Limit analysis to the final 40 meters:

```bash
uv run analyze_landings.py \
    --label expert \
    --min-altitude 0 \
    --max-altitude 40
```

Analyze only selected variables:

```bash
uv run analyze_landings.py \
    --label expert \
    --variables \
        airspeed_mps \
        aoa_deg \
        sideslip_deg \
        descent_rate_mps
```

Use custom directories:

```bash
uv run analyze_landings.py \
    --label fail \
    --input data/landing_cache/fail \
    --output data/landing_analysis/fail
```

## Analysis Workflow

The recommended workflow is:

```text
ArduPilot .BIN
      ↓
cache_landing_descents.py
      ↓
Landing Parquet files
      ↓
analyze_landings.py
      ↓
Altitude-binned statistics
      ↓
Expert vs Fail comparison
```

For comparison work, always run the expert and fail datasets using the same:

```text
--min-altitude
--max-altitude
--bin-width
```

so the resulting statistical distributions are directly comparable.
