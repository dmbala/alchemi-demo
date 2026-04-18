"""Shared Slurm/Singularity launcher helpers for all ALCHEMI pipelines."""

import argparse
import pathlib
import re
import subprocess
import sys

SIF = (
    "/n/holylfs06/LABS/kempner_shared/Everyone/containers/"
    "applications/alchemi-ht/alchemi_MACE.sif"
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


def add_common_args(p: argparse.ArgumentParser) -> None:
    """Flags every pipeline launcher shares."""
    p.add_argument("--partition", default="kempner_dev")
    p.add_argument("--account", default="kempner_dev")
    p.add_argument("--time", default="02:00:00")
    p.add_argument("--mem", default="32G")
    p.add_argument("--cpus", type=int, default=4)
    p.add_argument("--sif", default=SIF)
    p.add_argument("--project-root", default=pathlib.Path.cwd(), type=pathlib.Path)
    p.add_argument("--dry-run", action="store_true")


def sbatch_script(*, job_name, args, directives, bind, command, logs_dir) -> str:
    """Assemble an sbatch script. `directives` are extra #SBATCH lines (array, dependency, ...).

    `bind` is the full `--bind ...` string for `singularity exec`. `command` is the line
    after the container path — typically `python /pipeline/worker_xxx.py ...`.
    """
    root = args.project_root.resolve()
    is_array = any(d.startswith("--array=") for d in directives)
    output = logs_dir / ("job_%A_%a.out" if is_array else "job_%j.out")
    header = "\n".join(
        f"#SBATCH {d}" for d in (
            f"--job-name={job_name}",
            *directives,
            f"--partition={args.partition}",
            f"--account={args.account}",
            f"--cpus-per-task={args.cpus}",
            f"--mem={args.mem}",
            f"--time={args.time}",
            f"--output={output}",
        )
    )
    return f"""#!/bin/bash
{header}

set -euo pipefail
cd {root}

singularity exec --nv \\
    {SSL_FLAGS} \\
    {bind} \\
    {args.sif} \\
    {command}
"""


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
