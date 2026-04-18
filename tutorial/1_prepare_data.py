"""Chunk a GDB-17 SMILES file into CSVs for the Slurm array."""

import argparse
import csv
import gzip
import pathlib
import sys

DEFAULT_INPUT = (
    "/n/holylfs06/LABS/kempner_shared/Everyone/containers/"
    "applications/alchemi-ht/data/GDB17.50000000LLnoSR.smi.gz"
)


def open_smi(path: pathlib.Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunk a GDB-17 .smi(.gz) file.")
    p.add_argument("--input", default=DEFAULT_INPUT, type=pathlib.Path)
    p.add_argument("--outdir", default=pathlib.Path("data/chunks"), type=pathlib.Path)
    p.add_argument("--chunk-size", type=int, default=5000)
    p.add_argument("--max-molecules", type=int, default=None)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    chunk_idx = 0
    out_fh = writer = None

    with open_smi(args.input) as src:
        for line in src:
            parts = line.split()
            if not parts:
                continue
            smiles = parts[0]

            # Open a new chunk lazily when the previous one is full (or on the first row).
            if n_written % args.chunk_size == 0:
                if out_fh is not None:
                    out_fh.close()
                chunk_idx += 1
                out_fh = (args.outdir / f"chunk_{chunk_idx}.csv").open("w", newline="")
                writer = csv.writer(out_fh)
                writer.writerow(["smiles", "source_id"])

            source_id = parts[1] if len(parts) > 1 else f"{chunk_idx}-{n_written}"
            writer.writerow([smiles, source_id])
            n_written += 1

            if args.max_molecules is not None and n_written >= args.max_molecules:
                break

    if out_fh is not None:
        out_fh.close()

    if n_written == 0:
        print("ERROR: no molecules parsed from input", file=sys.stderr)
        return 1

    print(f"molecules={n_written} chunks={chunk_idx} outdir={args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
