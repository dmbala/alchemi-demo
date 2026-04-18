"""Solid-state pipeline launcher — one Slurm task per crystal.

Accepts either a list of --structures (names passed to ase.build.bulk) or a list
of --cifs (paths). One array task maps to one entry.
"""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _lib import add_common_args, build_gpu_array_launch, submit


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Launch solid-state EOS array.")
    p.add_argument("--structures", nargs="+", default=["Si", "Cu", "NaCl"])
    p.add_argument("--cifs", nargs="+", type=pathlib.Path, default=None,
                   help="Alternative to --structures: list of CIF paths, one per task.")
    p.add_argument("--output-dir", default=here / "data/outputs", type=pathlib.Path)
    p.add_argument("--logs-dir", default=here / "logs", type=pathlib.Path)
    p.add_argument("--concurrency", type=int, default=8)
    add_common_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    root = args.project_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.logs_dir.mkdir(parents=True, exist_ok=True)

    if args.cifs:
        entries = [f"'--cif /project/{p.resolve().relative_to(root)}'" for p in args.cifs]
    else:
        entries = [f"'--structure {s}'" for s in args.structures]
    arr_literal = " ".join(entries)
    n_tasks = len(entries)

    command = (
        f"bash -c \"ENTRIES=({arr_literal}); "
        "ARGS=\\${ENTRIES[\\$SLURM_ARRAY_TASK_ID-1]}; "
        "python /project/pipelines/4_solid_state/worker_solid_state.py "
        "--task-id \\$SLURM_ARRAY_TASK_ID "
        f"--output-dir /project/{args.output_dir.relative_to(root)} "
        "\\$ARGS\""
    )
    script = build_gpu_array_launch(
        job_name="alchemi_eos", args=args, n_tasks=n_tasks,
        concurrency=args.concurrency, command=command,
    )
    print(f"submitting solid-state array 1-{n_tasks}%{args.concurrency}")
    print(f"jobid={submit(script, args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
