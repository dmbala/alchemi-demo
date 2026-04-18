"""Catalysis pipeline launcher — one Slurm task per system (each task screens N sites)."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _lib import add_common_args, build_gpu_array_launch, submit


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Launch catalysis adsorption screen.")
    p.add_argument("--n-tasks", type=int, default=1)
    p.add_argument("--n-sites", type=int, default=20)
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

    command = (
        "python /project/pipelines/3_catalysis/worker_catalysis.py "
        "--task-id ${SLURM_ARRAY_TASK_ID} "
        f"--output-dir /project/{args.output_dir.relative_to(root)} "
        f"--n-sites {args.n_sites}"
    )
    script = build_gpu_array_launch(
        job_name="alchemi_catalysis", args=args, n_tasks=args.n_tasks,
        concurrency=args.concurrency, command=command,
    )
    print(f"submitting catalysis array 1-{args.n_tasks}%{args.concurrency}")
    print(f"jobid={submit(script, args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
