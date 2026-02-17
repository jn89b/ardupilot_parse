import argparse
import glob

from FlightParser import process_one_log, batch_export_logs

"""
- Parses ArduPilot BIN logs with pymavlink DFReader
- Exports a uniform-sampled JSON timeseries
- Uses EKF NED position from NKF1/NKF2/NKF3: PN/PE/PD (meters)
- Multiprocesses across MULTIPLE log files (recommended pattern)

NEW:
- Optional time slicing: write one JSON per time window (e.g., every 60 seconds)
  Example output names:
    <base>_part000.json, <base>_part001.json, ...

Notes:
- Do NOT share FlightParser / DFReader across processes.
- Each worker opens its own BIN file and writes its own JSON.

Usage examples:
  python generate_json.py --log data/example.BIN --out data/example_export.json
  python generate_json.py --log data/example.BIN --out data/example_export.json --slice_seconds 60

  python generate_json.py --glob "data/logs/**/*.BIN" --out_dir data/json_out --workers 8
  python generate_json.py --glob "data/logs/**/*.BIN" --out_dir data/json_out --workers 8 --slice_seconds 60
"""


# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=str, default=None, help="Single BIN log path")
    ap.add_argument("--out", type=str, default=None, help="Output JSON path for --log")

    ap.add_argument("--glob", type=str, default=None, help='Glob for multiple logs, e.g. "data/logs/**/*.BIN"')
    ap.add_argument("--out_dir", type=str, default="data/json_out", help="Output directory for batch mode")

    ap.add_argument("--dt", type=float, default=0.05, help="Output sample period (seconds)")
    ap.add_argument("--slice_seconds", type=float, default=None,
                    help="If set, split output into multiple JSONs each covering this many seconds (e.g. 60)")

    ap.add_argument("--workers", type=int, default=8, help="Number of processes for batch mode")
    ap.add_argument("--strict_nkf", action="store_true", help="Error if NKF PN/PE/PD missing")
    ap.add_argument("--verbose", action="store_true", help="Print message types present per log")

    args = ap.parse_args()

    if args.log:
        if not args.out:
            raise SystemExit("--out is required when using --log")

        out_paths = process_one_log((
            args.log,
            args.out,
            args.dt,
            args.strict_nkf,
            args.verbose,
            args.slice_seconds,   # NEW
        ))

        # process_one_log now returns List[str]
        if isinstance(out_paths, list):
            print("Wrote:", len(out_paths), "files")
            for p in out_paths:
                print(" -", p)
        else:
            # fallback if your FlightParser hasn't been updated yet
            print("Wrote:", out_paths)
        return

    if args.glob:
        paths = sorted(glob.glob(args.glob, recursive=True))
        if not paths:
            raise SystemExit(f"No logs matched glob: {args.glob}")

        outs = batch_export_logs(
            log_paths=paths,
            out_dir=args.out_dir,
            dt=args.dt,
            workers=args.workers,
            strict_nkf=args.strict_nkf,
            verbose=args.verbose,
            slice_seconds=args.slice_seconds,  # NEW
        )

        print("Wrote", len(outs), "files to", args.out_dir)
        return

    raise SystemExit("Provide either --log/--out or --glob/--out_dir")


if __name__ == "__main__":
    main()
