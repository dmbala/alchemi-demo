# ALCHEMI HT tutorial

A hands-on walkthrough of a 4-phase, GPU-accelerated molecular screening pipeline on FASRC Slurm. You'll go from raw SMILES strings to filtered "stable" molecules with HOMO/LUMO gaps, running entirely inside a shared Singularity container.

Pipeline at a glance:

```
SMILES  ->  3D coords     ->  relaxed geometry   ->  electronic properties
          (RDKit ETKDGv3)   (AIMNet2 + ASE FIRE)     (gpu4pyscf DFT)
```

Each phase is one small Python script. No static `.slurm` files — Phase 2 composes the Slurm batch script in Python and pipes it to `sbatch`.

## What you'll need

- FASRC account with a Kempner Slurm allocation (this tutorial defaults to `--account=kempner_dev --partition=kempner_dev`).
- The shared container at `/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/alchemi-ht/alchemi_MACE.sif` — already built, nothing to install.
- Pre-staged AIMNet2 weights at `/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/alchemi/container/aimnet_assets/aimnet2_wb97m_d3_0.pt`.
- `python3.12` on the head node. **Not** `python3` — the system default is 3.6 and won't parse modern syntax.

## The four scripts

| File | Runs on | What it does |
| --- | --- | --- |
| `1_prepare_data.py` | Head node | Streams a `.smi(.gz)` file, splits it into `data/chunks/chunk_N.csv` |
| `2_launch_slurm.py` | Head node | Submits the GPU array + chains Phase 4 via `--dependency=afterany` |
| `3_worker_node.py` | GPU compute node (inside the container) | Embed → relax → DFT for every row in one chunk |
| `4_aggregate_and_learn.py` | Compute node (auto-triggered) | Concatenates per-chunk results, filters by energy/gap |

## Step 1 — Smoke test (tiny, <2 minutes of GPU time)

Work from the project root (`alchemi-demo/`). First, chunk 4 molecules from the default GDB-17 subset:

```bash
python3.12 tutorial/1_prepare_data.py --max-molecules 4 --chunk-size 2
# -> molecules=4 chunks=2 outdir=data/chunks
```

You should see two files:

```bash
ls data/chunks
# chunk_1.csv  chunk_2.csv
```

Each chunk is a tiny CSV:

```csv
smiles,source_id
BrC1=C2C3=C4C(CC3CCC2=O)C(=N)NC4=N1,1-0
BrC1=C2C3C4CCC(C4)C3C(=N)OC2=NC=C1,1-1
```

## Step 2 — Inspect what Phase 2 will submit

Before touching the queue, ask the launcher to render its sbatch scripts without submitting:

```bash
python3.12 tutorial/2_launch_slurm.py --dry-run
```

Two sbatch scripts are printed. Note two things:

1. The worker array binds `data/`, `tutorial/`, and the `aimnet_assets` directory, and passes the three SSL-cert env vars so anything inside the container that opens an HTTPS connection uses the host certificate bundle.
2. The aggregator is submitted with `--dependency=afterany:<worker_jobid>` — Slurm itself releases it only after every array task reaches a terminal state.

## Step 3 — Submit for real

```bash
python3.12 tutorial/2_launch_slurm.py
# found 2 chunks; submitting array 1-2%16 to kempner_dev
# Submitted batch job 6467462
# Submitted batch job 6467463
# array_jobid=6467462 phase4_jobid=6467463
```

Watch the queue:

```bash
squeue -u $USER -o "%i %j %T %M %R"
```

You'll see the aggregator parked in `PENDING (Dependency)` until both array tasks finish. Each worker takes roughly 1–2 minutes per molecule for B3LYP/def2-SVP on an H100/H200 — plan accordingly for larger runs.

## Step 4 — Read the results

When both jobs leave the queue:

```bash
cat data/results/stable.csv
# smiles,energy,homo,lumo,gap,n_atoms,source_id
# BrC1=C2C3=C4C(CC3CCC2=O)C(=N)NC4=N1,-3276.09...,-6.36,-1.85,4.51,27,1-0
# ...
```

Columns:
- `energy` in Hartree (total DFT electronic energy).
- `homo` / `lumo` / `gap` in eV. Gap is a proxy for kinetic stability — larger = more chemically inert.
- `n_atoms` after RDKit adds hydrogens.

`logs/phase4_summary.txt` has a one-line audit:

```
timestamp=...Z files=2 total=4 stable=4 energy_max=0.0 gap_min=1.0
```

## Step 5 — Scale up

For a full screen, drop `--max-molecules` and keep the default `--chunk-size 5000`:

```bash
python3.12 tutorial/1_prepare_data.py
python3.12 tutorial/2_launch_slurm.py --concurrency 16
```

`--concurrency` caps simultaneously-running array tasks. Raise it when the queue is cold, drop it when the partition is busy.

## Gotchas baked into the code

You won't hit these yourself, but it's worth knowing why the code looks the way it does:

- **AIMNet2 weights are bind-mounted** to `/aimnet_assets` because the SIF is read-only. If `AIMNet2Calculator("aimnet2")` ran with no explicit model path, it would try to create an `assets/` directory inside the container's install tree and fail with `Read-only file system`. We pre-stage the `.pt` and pass its absolute in-container path.
- **Dispersion is disabled** (`needs_dispersion=False`). The container's `aimnet` + `nvalchemiops._dftd3` dispersion shim has a signature mismatch (`TypeError: dftd3() got an unexpected keyword argument 'cell'`). Skipping the external D3 correction loses <1 kcal/mol of accuracy — negligible for HT screening. Revisit if you need publication-grade absolute energies.
- **SSL env vars** (`SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `REQUESTS_CA_BUNDLE`) are pointed at the host's cert bundle so any network call inside the container works (HF downloads, pyscf basis fetches, etc.).
- **Harmless startup noise:** `rm: cannot remove '/usr/local/cuda/compat/lib': Read-only file system` prints every time the container starts — ignore it.

## Prefer notebooks?

Every phase has a companion notebook in `tutorial/notebooks/`:

| Notebook | Kernel location | Purpose |
| --- | --- | --- |
| `01_prepare_data.ipynb` | Head node, `python3.12` | Chunking demo with inline pandas inspection |
| `02_launch_slurm.ipynb` | Head node, `python3.12` | Dry-run inspection, submit, poll `squeue`, `sacct` check |
| `03_worker_demo.ipynb` | **Inside the container, on a GPU node** | Step-by-step physics for a single molecule (RDKit → AIMNet2 → gpu4pyscf) |
| `04_aggregate_and_analyze.ipynb` | Head node, `python3.12` + matplotlib | Concatenate, plot gap distribution, apply filter |

Notebook 03 needs `jupyter lab` running *inside* `alchemi_MACE.sif` on a GPU allocation — the notebook's header has the exact `salloc` + `singularity exec` incantation.

## Where to go next

- Swap `--basis def2-svp` for `sto-3g` in `3_worker_node.py` (via `--basis`) for a cheap smoke-test basis.
- Edit the filter in `4_aggregate_and_learn.py` (`--energy-max`, `--gap-min`) to match your stability criteria.
- Fill in the `retrain_surrogate(df)` stub at the bottom of Phase 4 to close the active-learning loop on the newly-labeled rows.

Each script is ~50–150 lines; read them top-to-bottom — they're meant to be legible.
