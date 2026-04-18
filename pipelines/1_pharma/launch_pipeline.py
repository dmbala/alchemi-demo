"""Pharma pipeline launcher — one Slurm task per molecule in the input CSV."""

import argparse
import csv
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _lib import add_common_args, sbatch_script, submit, AIMNET_ASSETS


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Launch pharma conformer-search array.")
    p.add_argument("--input", default=here / "data/inputs/molecules.csv", type=pathlib.Path)
    p.add_argument("--output-dir", default=here / "data/outputs", type=pathlib.Path)
    p.add_argument("--logs-dir", default=here / "logs", type=pathlib.Path)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument("--aimnet-assets", default=AIMNET_ASSETS)
    add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    with args.input.open() as f:
        n_mols = sum(1 for _ in csv.DictReader(f))
    if n_mols == 0:
        print(f"ERROR: no molecules in {args.input}", file=sys.stderr)
        return 1

    root = args.project_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    directives = (
        f"--array=1-{n_mols}%{args.concurrency}",
        "--gres=gpu:1",
    )
    bind = (
        f"--bind {root}:/project "
        f"--bind {args.aimnet_assets}:/aimnet_assets:ro"
    )
    cmd = (
        "python /project/pipelines/1_pharma/worker_pharma.py "
        f"--input /project/{args.input.relative_to(root)} "
        "--task-id ${SLURM_ARRAY_TASK_ID} "
        f"--output-dir /project/{args.output_dir.relative_to(root)}"
    )
    script = sbatch_script(
        job_name="alchemi_pharma", args=args, directives=directives,
        bind=bind, command=cmd, logs_dir=args.logs_dir,
    )
    print(f"submitting pharma array 1-{n_mols}%{args.concurrency}")
    jobid = submit(script, args.dry_run)
    print(f"jobid={jobid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
