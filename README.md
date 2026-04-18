# alchemi-demo

GPU-accelerated computational-chemistry pipelines for the FASRC Kempner cluster, built on the [NVIDIA ALCHEMI](https://github.com/NVIDIA/nvalchemi-toolkit) toolkit.

Two worked examples live here, each aimed at a different audience:

| Directory | What it is | Audience |
| --- | --- | --- |
| [`tutorial/`](tutorial/) | A **4-phase high-throughput screening** pipeline (GDB-17 SMILES → 3D embed → AIMNet2 relax → gpu4pyscf DFT HOMO/LUMO). End-to-end Slurm + Singularity orchestration in Python. | Readers learning the **orchestration pattern** — how to chunk, submit, chain with `--dependency`, aggregate. |
| [`pipelines/`](pipelines/) | **Four independent domain demos** under one container: [pharma conformer search (AIMNet2)](pipelines/1_pharma/), [Li⁺ solvation MD (MACE)](pipelines/2_battery/), [CO₂/Cu catalysis (MACE)](pipelines/3_catalysis/), and [solid-state EOS (MACE)](pipelines/4_solid_state/). | Readers who want a **worked example in their science domain** and will adapt one of them. |

Both share the same shared container:

```
/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/alchemi-ht/alchemi_MACE.sif
```

The container layers `mace-torch` onto the earlier `alchemi_ht.sif` (which itself inherits NVIDIA's NGC PyTorch + `nvalchemi-toolkit` base). Build recipe: [`container/alchemi_MACE.def`](container/alchemi_MACE.def).

## Quickstart

From the repo root with `python3.12` on the head node (system `python3` is 3.6):

```bash
# HT screening walkthrough (10-molecule smoke test)
python3.12 tutorial/1_prepare_data.py --max-molecules 10 --chunk-size 5
python3.12 tutorial/2_launch_slurm.py                    # submits array + dep aggregator

# OR one of the four domain pipelines (pharma is simplest)
python3.12 pipelines/1_pharma/launch_pipeline.py --dry-run
python3.12 pipelines/1_pharma/launch_pipeline.py
```

Every launcher accepts `--dry-run` — inspect the generated sbatch script before it goes to the queue.

## What's where

```
alchemi-demo/
├── README.md                    # (this file) — routing
├── container/
│   └── alchemi_MACE.def         # Singularity recipe for the shared SIF
├── tutorial/
│   ├── 1_prepare_data.py        # Phase 1: chunk SMILES
│   ├── 2_launch_slurm.py        # Phase 2: Python-native sbatch launcher
│   ├── 3_worker_node.py         # Phase 3: per-chunk GPU worker (inside container)
│   ├── 4_aggregate_and_learn.py # Phase 4: stream-filter results, flag stable
│   ├── README.md                # full walkthrough
│   └── notebooks/               # companion Jupyter tutorials (4 notebooks)
├── pipelines/
│   ├── README.md                # per-pipeline deep dive
│   ├── _lib.py                  # shared sbatch composer + submit helper
│   ├── 1_pharma/                # conformer search worker + launcher
│   ├── 2_battery/               # Li⁺ MD worker + launcher
│   ├── 3_catalysis/             # CO₂/Cu(111) adsorption worker + launcher
│   └── 4_solid_state/           # EOS → bulk modulus worker + launcher
└── notes/
    ├── notes.md                 # original HT-screening spec
    └── pipeline_notes.md        # four-pipeline workshop spec
```

## Where to start

- **Brand-new to Slurm+Singularity orchestration?** Read [`tutorial/README.md`](tutorial/README.md) top-to-bottom. Its four phases walk through the whole pattern in ~20 minutes of reading.
- **Already know HPC, want a domain example?** Jump to [`pipelines/README.md`](pipelines/README.md) and pick the pipeline closest to your science.
- **Building your own pipeline from scratch?** Copy one of the `pipelines/*/` dirs as a template — each is ~200 lines of self-contained Python plus a thin launcher that reuses `pipelines/_lib.py`.

## Environment assumptions

- **Slurm account/partition:** defaults to `--account=kempner_dev --partition=kempner_dev`. Override with the launcher flags. Check your own access with `sacctmgr show assoc user=$USER`.
- **AIMNet2 weights:** pre-staged at `/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/alchemi/container/aimnet_assets/aimnet2_wb97m_d3_0.pt`. Used by `tutorial/` and `pipelines/1_pharma/`. Bind-mounted into the container at runtime.
- **MACE-MP-0 weights:** auto-downloaded to `~/.cache/mace/` on first use. No pre-staging.
- **Head-node Python:** use `python3.12` (or `module load python/3.12.x-fasrc01`). The launchers use modern syntax that 3.6 won't parse.

## Relationship between the two directories

`tutorial/` and `pipelines/` are independent. Neither imports from the other. If you only care about one, ignore the other.

A practical path: read `tutorial/` to understand the orchestration pattern, then use one of the `pipelines/` as a starting point for your own domain science.
