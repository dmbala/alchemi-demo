"""Battery pipeline launcher — submits N independent MD runs (array) with different seeds."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _lib import add_common_args, sbatch_script, submit


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Launch battery MD array.")
    p.add_argument("--n-tasks", type=int, default=1,
                   help="How many independent MD runs to spawn (different random seeds).")
    p.add_argument("--n-steps", type=int, default=1000)
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

    directives = (
        f"--array=1-{args.n_tasks}%{args.concurrency}",
        "--gres=gpu:1",
    )
    bind = f"--bind {root}:/project"
    cmd = (
        "python /project/pipelines/2_battery/worker_battery.py "
        "--task-id ${SLURM_ARRAY_TASK_ID} "
        f"--output-dir /project/{args.output_dir.relative_to(root)} "
        f"--n-steps {args.n_steps}"
    )
    script = sbatch_script(
        job_name="alchemi_battery", args=args, directives=directives,
        bind=bind, command=cmd, logs_dir=args.logs_dir,
    )
    print(f"submitting battery array 1-{args.n_tasks}%{args.concurrency}")
    jobid = submit(script, args.dry_run)
    print(f"jobid={jobid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
