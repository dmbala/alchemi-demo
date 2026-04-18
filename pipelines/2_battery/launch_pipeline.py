"""Battery pipeline launcher — submits N independent MD runs (array) with different seeds."""

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from _lib import add_common_args, build_gpu_array_launch, submit


def parse_args() -> argparse.Namespace:
    here = pathlib.Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description="Launch battery MD array.")
    p.add_argument("--n-tasks", type=int, default=1)
    p.add_argument("--n-steps", type=int, default=1000)
    p.add_argument("--initial-box", type=pathlib.Path, default=None,
                   help="Optional initial box (xyz/extxyz). Forwarded to every task.")
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

    initial_box_flag = ""
    if args.initial_box is not None:
        initial_box = args.initial_box.expanduser().resolve()
        try:
            initial_box_in_project = initial_box.relative_to(root)
        except ValueError as exc:
            raise SystemExit(
                f"--initial-box must be inside --project-root so it is reachable "
                f"inside the container: {initial_box} is not under {root}"
            ) from exc
        initial_box_flag = f"--initial-box /project/{initial_box_in_project} "
    command = (
        "python /project/pipelines/2_battery/worker_battery.py "
        "--task-id ${SLURM_ARRAY_TASK_ID} "
        f"--output-dir /project/{args.output_dir.relative_to(root)} "
        f"--n-steps {args.n_steps} "
        f"{initial_box_flag}"
    )
    script = build_gpu_array_launch(
        job_name="alchemi_battery", args=args, n_tasks=args.n_tasks,
        concurrency=args.concurrency, command=command,
    )
    print(f"submitting battery array 1-{args.n_tasks}%{args.concurrency}")
    print(f"jobid={submit(script, args.dry_run)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
