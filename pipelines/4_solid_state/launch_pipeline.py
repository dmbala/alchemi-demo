"""Solid-state pipeline launcher — one Slurm task per crystal."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _lib import add_common_args, sbatch_script, submit


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Launch solid-state EOS array.")
    p.add_argument("--structures", nargs="+", default=["Si", "Cu", "NaCl"],
                   help="Structures to pass to ase.build.bulk, one task each.")
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

    # Lay out structures as a bash array so the sbatch body can index it.
    structures_arr = " ".join(f"'{s}'" for s in args.structures)
    n_tasks = len(args.structures)

    directives = (
        f"--array=1-{n_tasks}%{args.concurrency}",
        "--gres=gpu:1",
    )
    bind = f"--bind {root}:/project"
    cmd = (
        f"bash -c \"STRUCTURES=({structures_arr}); "
        "STRUCTURE=\\${STRUCTURES[\\$SLURM_ARRAY_TASK_ID-1]}; "
        "python /project/pipelines/4_solid_state/worker_solid_state.py "
        "--task-id \\$SLURM_ARRAY_TASK_ID "
        "--structure \\$STRUCTURE "
        f"--output-dir /project/{args.output_dir.relative_to(root)}\""
    )
    script = sbatch_script(
        job_name="alchemi_eos", args=args, directives=directives,
        bind=bind, command=cmd, logs_dir=args.logs_dir,
    )
    print(f"submitting solid-state array 1-{n_tasks}%{args.concurrency}")
    jobid = submit(script, args.dry_run)
    print(f"jobid={jobid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
