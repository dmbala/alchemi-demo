"""Compose sbatch scripts and submit the worker array + dependent aggregator."""

import argparse
import pathlib
import re
import subprocess
import sys

SIF = (
    "/n/holylfs06/LABS/kempner_shared/Everyone/containers/"
    "applications/alchemi-ht/alchemi_ht.sif"
)
AIMNET_ASSETS = (
    "/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/alchemi/container/aimnet_assets"
)
SSL_FLAGS = (
    "--env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt "
    "--env CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt "
    "--env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt"
)

JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Launch ALCHEMI HT Slurm array.")
    p.add_argument("--chunks-dir", default=pathlib.Path("data/chunks"), type=pathlib.Path)
    p.add_argument("--results-dir", default=pathlib.Path("data/results"), type=pathlib.Path)
    p.add_argument("--logs-dir", default=pathlib.Path("logs"), type=pathlib.Path)
    p.add_argument("--partition", default="kempner_dev")
    p.add_argument("--account", default="kempner_dev")
    p.add_argument("--time", default="04:00:00")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--concurrency", type=int, default=16,
                   help="Max simultaneously running array tasks (sbatch %N).")
    p.add_argument("--sif", default=SIF)
    p.add_argument("--aimnet-assets", default=AIMNET_ASSETS,
                   help="Host dir with pre-staged AIMNet2 .pt files; mounted at /aimnet_assets.")
    p.add_argument("--project-root", default=pathlib.Path.cwd(), type=pathlib.Path)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def sbatch_script(job_name, directives, args, container_cmd, extra_binds="") -> str:
    root = args.project_root.resolve()
    is_array = any(d.startswith("--array=") for d in directives)
    output = args.logs_dir / ("job_%A_%a.out" if is_array else "phase4_%j.out")
    header = "\n".join(
        f"#SBATCH {d}" for d in (
            f"--job-name={job_name}",
            *directives,
            f"--partition={args.partition}",
            f"--account={args.account}",
            f"--output={output}",
        )
    )
    binds = f"--bind {root}/data:/data --bind {root}/tutorial:/tutorial {extra_binds}".strip()
    return f"""#!/bin/bash
{header}

set -euo pipefail
cd {root}

singularity exec --nv \\
    {SSL_FLAGS} \\
    {binds} \\
    {args.sif} \\
    {container_cmd}
"""


def build_array_script(args, n_chunks: int) -> str:
    directives = (
        f"--array=1-{n_chunks}%{args.concurrency}",
        "--gres=gpu:1",
        f"--cpus-per-task={args.cpus}",
        f"--mem={args.mem}",
        f"--time={args.time}",
    )
    cmd = (
        "python /tutorial/3_worker_node.py "
        "--input /data/chunks/chunk_${SLURM_ARRAY_TASK_ID}.csv "
        "--output /data/results/result_${SLURM_ARRAY_TASK_ID}.csv"
    )
    return sbatch_script(
        "alchemi_batch", directives, args, cmd,
        extra_binds=f"--bind {args.aimnet_assets}:/aimnet_assets:ro",
    )


def build_phase4_script(args, dep_jobid: str) -> str:
    directives = (
        f"--dependency=afterany:{dep_jobid}",
        "--cpus-per-task=2",
        "--mem=8G",
        "--time=00:30:00",
    )
    root = args.project_root.resolve()
    cmd = (
        "python /tutorial/4_aggregate_and_learn.py "
        "--results-dir /data/results "
        "--output /data/results/stable.csv "
        "--summary /logs/phase4_summary.txt"
    )
    return sbatch_script(
        "alchemi_agg", directives, args, cmd,
        extra_binds=f"--bind {root}/logs:/logs",
    )


def submit(script: str, dry_run: bool) -> str:
    if dry_run:
        print("---DRY RUN sbatch script---")
        print(script)
        return "DRYRUN"
    proc = subprocess.run(["sbatch"], input=script, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"sbatch stderr: {proc.stderr.strip()}", file=sys.stderr)
        print(f"sbatch stdout: {proc.stdout.strip()}", file=sys.stderr)
        raise RuntimeError(f"sbatch failed (rc={proc.returncode})")
    print(proc.stdout.strip())
    m = JOBID_RE.search(proc.stdout)
    if not m:
        raise RuntimeError(f"could not parse jobid from: {proc.stdout!r}")
    return m.group(1)


def main() -> int:
    args = parse_args()
    chunks = sorted(args.chunks_dir.glob("chunk_*.csv"))
    if not chunks:
        print(f"ERROR: no chunks in {args.chunks_dir}", file=sys.stderr)
        return 1

    n_chunks = len(chunks)
    print(f"found {n_chunks} chunks; submitting array 1-{n_chunks}%{args.concurrency} to {args.partition}")

    array_jobid = submit(build_array_script(args, n_chunks), args.dry_run)
    phase4_jobid = submit(build_phase4_script(args, array_jobid), args.dry_run)
    print(f"array_jobid={array_jobid} phase4_jobid={phase4_jobid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
