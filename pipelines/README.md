# ALCHEMI pipelines

Four domain-specific high-throughput pipelines on the Kempner Slurm cluster, all backed by the shared container at `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/alchemi-ht/alchemi_MACE.sif`.

| # | Pipeline | Physics | One task produces |
| --- | --- | --- | --- |
| 1 | [`1_pharma/`](1_pharma/) | RDKit ETKDG (500 confs) → AIMNet2 relax → rank/RMSD | top-5 lowest-energy distinct conformers (extxyz) |
| 2 | [`2_battery/`](2_battery/) | MACE-MP-0 NVT Langevin MD → Li–O RDF | RDF JSON + final frame |
| 3 | [`3_catalysis/`](3_catalysis/) | Cu(111) + CO₂ × N sites → MACE relax → E_ads | ranked adsorption CSV + per-site xyz |
| 4 | [`4_solid_state/`](4_solid_state/) | Strain cell 90–110% → MACE single-point → Birch–Murnaghan fit | bulk modulus JSON |

Each pipeline directory contains:

- `worker_<domain>.py` — the physics (runs inside the container on a GPU node)
- `launch_pipeline.py` — Slurm array launcher (runs on the head node)
- `data/inputs/` — sample inputs (where needed)
- `data/outputs/` — per-task results land here
- `logs/` — Slurm stdout/stderr

Shared infrastructure lives in [`_lib.py`](_lib.py): the container path, SSL env flags, AIMNet2 asset dir, and the `sbatch` composition helper.

## Quickstart

```bash
# from the repo root (alchemi-demo/)
python3.12 pipelines/1_pharma/launch_pipeline.py --dry-run           # inspect sbatch script
python3.12 pipelines/1_pharma/launch_pipeline.py                     # submit

python3.12 pipelines/4_solid_state/launch_pipeline.py --structures Si  # single EOS
python3.12 pipelines/4_solid_state/launch_pipeline.py                  # Si + Cu + NaCl
```

All launchers accept:
- `--partition / --account` (defaults `kempner_dev`)
- `--time / --mem / --cpus` (sbatch overrides)
- `--concurrency` (array throttle `%N`)
- `--dry-run` (print sbatch, don't submit)

## When to run which

- **Pharma** — only pipeline that doesn't need MACE; works with the existing AIMNet2 weights. Start here if you want to validate the Slurm + container plumbing on a known-good physics stack.
- **Solid-state** — simplest MACE pipeline (single-point only, no relaxation). Quickest MACE smoke test.
- **Catalysis** — long serial relaxation loop per task; scale down `--n-sites` for fast iteration.
- **Battery** — MD is the slowest by far. Keep `--n-steps 1000` for smoke tests; the spec's 10,000-step target takes minutes per task.
