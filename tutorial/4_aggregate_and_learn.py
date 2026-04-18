"""Concatenate per-chunk results, filter for stability."""

import argparse
import pathlib
import sys
from datetime import datetime, timezone


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate ALCHEMI HT results.")
    p.add_argument("--results-dir", default=pathlib.Path("data/results"), type=pathlib.Path)
    p.add_argument("--output", default=pathlib.Path("data/results/stable.csv"), type=pathlib.Path)
    p.add_argument("--summary", default=pathlib.Path("logs/phase4_summary.txt"), type=pathlib.Path)
    p.add_argument("--energy-max", type=float, default=0.0,
                   help="Keep rows with energy < energy_max (Hartree).")
    p.add_argument("--gap-min", type=float, default=1.0,
                   help="Keep rows with gap > gap_min (eV).")
    return p.parse_args()


def main() -> int:
    import pandas as pd

    args = parse_args()
    paths = sorted(args.results_dir.glob("result_*.csv"))
    if not paths:
        print(f"ERROR: no result files in {args.results_dir}", file=sys.stderr)
        return 1

    # Stream per file so RAM stays flat for arbitrarily large runs.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_total = n_stable = 0
    write_header = True
    with args.output.open("w", newline="") as fout:
        for p in paths:
            chunk = pd.read_csv(p)
            n_total += len(chunk)
            keep = chunk[(chunk["energy"] < args.energy_max) & (chunk["gap"] > args.gap_min)]
            n_stable += len(keep)
            keep.to_csv(fout, index=False, header=write_header)
            write_header = False

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        f"timestamp={datetime.now(timezone.utc).isoformat()} "
        f"files={len(paths)} total={n_total} stable={n_stable} "
        f"energy_max={args.energy_max} gap_min={args.gap_min}\n"
    )
    print(
        f"aggregated files={len(paths)} total={n_total} "
        f"stable={n_stable} -> {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
