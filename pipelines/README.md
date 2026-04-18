# ALCHEMI workshop pipelines

Four domain-specific, GPU-accelerated computational-chemistry pipelines running on the Kempner Slurm cluster, all backed by one shared Singularity container:

```
/n/holylfs06/LABS/kempner_shared/Everyone/containers/applications/alchemi-ht/alchemi_MACE.sif
```

Each pipeline is a small, self-contained domain demo — pick the one closest to your science, read its ~150-line worker, change the input, scale up.

| # | Pipeline | Model(s) | Task unit | Output |
| --- | --- | --- | --- | --- |
| 1 | [**Pharma**](1_pharma/) — conformer search | AIMNet2 | 1 SMILES → 500 conformers | top-k lowest-energy distinct conformers (`.xyz`) |
| 2 | [**Battery**](2_battery/) — Li⁺ solvation MD | MACE-MP-0 | 1 simulation box → 10 ps NVT | Li–O RDF JSON + final frame |
| 3 | [**Catalysis**](3_catalysis/) — CO₂/Cu(111) adsorption | MACE-MP-0 | 1 system → N binding sites | ranked `E_ads` CSV + per-site `.xyz` |
| 4 | [**Solid-state**](4_solid_state/) — Equation of State | MACE-MP-0 | 1 crystal → 20 strained volumes | bulk modulus JSON + relaxed `.xyz` |

## Layout

```
pipelines/
├── _lib.py                       # shared sbatch composer + submit helper
├── README.md                     # (this file)
├── 1_pharma/
│   ├── launch_pipeline.py        # head-node Slurm launcher
│   ├── worker_pharma.py          # compute-node physics (inside container)
│   ├── data/inputs/molecules.csv # sample SMILES: aspirin, ibuprofen, caffeine
│   └── data/outputs/             # per-task xyz files (gitignored)
├── 2_battery/                    # same shape, no input file (box built in-worker)
├── 3_catalysis/                  # same shape, slab + adsorbate generated in-worker
└── 4_solid_state/                # same shape, bulk crystals via ase.build.bulk()
```

Each `worker_*.py` is a CLI tool callable standalone for ad-hoc runs; each `launch_pipeline.py` is the thin wrapper that submits one Slurm array where each task invokes the worker.

## One-time prerequisites

1. **FASRC account** with Slurm access to `kempner_dev` partition (or override with `--partition / --account`).
2. **`python3.12`** on the head node (system `python3` is 3.6 on FASRC and will not parse modern syntax).
3. **AIMNet2 weights** pre-staged at `/n/netscratch/kempner_dev/Lab/bdesinghu/Agent/alchemi/container/aimnet_assets/aimnet2_wb97m_d3_0.pt`. Only pharma uses these. Override with `--aimnet-assets` if you staged them elsewhere.
4. **MACE-MP-0 weights** — *nothing to pre-stage*. The first worker to use MACE auto-downloads to `~/.cache/mace/` (user-writable even inside the read-only SIF). Subsequent workers hit the cache.

## Workflow for every pipeline

```bash
# from the repo root (alchemi-demo/)

# 1. dry-run — inspect the sbatch script that will be submitted
python3.12 pipelines/1_pharma/launch_pipeline.py --dry-run

# 2. submit
python3.12 pipelines/1_pharma/launch_pipeline.py

# 3. watch
squeue -u $USER -o "%i %j %T %M %R"

# 4. when done, look at outputs
ls pipelines/1_pharma/data/outputs/
```

All launchers share a common flag vocabulary (defined in `_lib.add_common_args`):

| Flag | Default | Purpose |
| --- | --- | --- |
| `--partition`, `--account` | `kempner_dev` / `kempner_dev` | Slurm routing |
| `--time` | `02:00:00` | sbatch `--time` |
| `--mem` | `32G` | sbatch `--mem` |
| `--cpus` | `4` | `--cpus-per-task` |
| `--sif` | shared MACE SIF | Singularity image path |
| `--concurrency` | `8` or `16` | array throttle (`%N`) |
| `--dry-run` | — | print sbatch script, don't submit |

## Pipeline 1 — Pharma

**Scientific question.** Flexible drug molecules have many accessible 3D conformations. Which one is lowest in energy? How much does it matter that I picked the "right" conformer?

**What happens in one task.**

1. Read one SMILES from `data/inputs/molecules.csv` (row = `--task-id`).
2. RDKit `EmbedMultipleConfs(ETKDGv3)` generates 500 distinct 3D conformers, followed by an MMFF94 cleanup.
3. AIMNet2 (via ASE `FIRE`) relaxes every conformer independently.
4. Rank by energy, drop anything within 0.5 Å RMSD of a higher-ranked structure (keeps the top-k *distinct*).
5. Write `<name>_top<k>.xyz` in extxyz format with per-conformer energy in the comment line.

**Key flags.** `--n-confs 500` (default), `--top-k 5`, `--rmsd-threshold 0.5`, `--fmax 0.05`.

**Smoke-test output.** Aspirin, 20 confs, top-3:
```
$ python3.12 pipelines/1_pharma/worker_pharma.py --input ... --task-id 1 --output-dir ... --n-confs 20 --top-k 3
embedded 20 conformers
relaxed 20/20 conformers
done top=3 min_energy=-17663.4226 eV elapsed=6.3s
```

## Pipeline 2 — Battery

**Scientific question.** How does a lithium ion behave in an electrolyte? How many solvent oxygens cluster around it? How does that coordination shell change with temperature?

**What happens in one task.**

1. Build (or load via `--initial-box`) a cubic box: 10 water molecules + 1 Li⁺, random non-overlapping insertion.
2. MACE-MP-0 attached as ASE calculator, device=`cuda`, float32.
3. Maxwell–Boltzmann velocity init at 300 K, then Langevin NVT (`friction = 0.01/fs`).
4. Run `--n-steps` MD steps (default 1000 for smoke tests; **spec target: 10,000**).
5. Record positions every `--sample-every` step; compute the **Li–O radial distribution function** over the trajectory.
6. Emit `task{N}_rdf_li_o.json` with `r` and `g(r)` arrays + a final-frame `.xyz`.

**Key flags.** `--n-steps`, `--temperature-k 300`, `--sample-every 10`, `--rdf-rmax 6.0`, `--rdf-nbins 60`, `--mace-model small|medium|large`.

**Smoke-test output.**
```
system size: n_atoms=31 n_Li=1 n_O=10
MD done: 100 steps in 10.3s (11 snapshots)
wrote RDF + final frame
```

**To scale up.** Swap the 10-water box for a real electrolyte (edit `build_default_box` or pass `--initial-box my_box.xyz`), push `--n-steps` to 10,000+.

## Pipeline 3 — Catalysis

**Scientific question.** CO₂ on a copper surface — where does it bind, how strongly, and does it dissociate? High-throughput screens need per-site adsorption energies across many candidate positions.

**What happens in one task.**

1. Build a Cu(111) slab with `ase.build.fcc111` (`--facet-size 3 3 --n-layers 4`).
2. Compute **reference energies**: relax the clean slab, relax isolated CO₂.
3. Generate `--n-sites` binding configurations by gridding xy positions above the top layer with a small random jitter (deterministic seed).
4. For each site: place CO₂ at `--initial-height` Å above the top Cu, freeze the bottom two slab layers, relax with MACE + FIRE.
5. Compute `E_ads = E_system − (E_slab + E_CO2)` and check for dissociation (max C–O bond > 2.0 Å).
6. Emit `task{N}_adsorption.csv` sorted by `E_ads`, plus per-site `.xyz` files.

**Key flags.** `--metal Cu`, `--facet-size 3 3`, `--n-layers 4`, `--n-sites 20`, `--fmax 0.05`, `--max-opt-steps 100`.

**Smoke-test output.** Cu(111) × 4 sites:
```
reference: clean slab
reference: isolated CO2
generating 4 sites and relaxing
done E_slab=-138.91 eV E_CO2=-22.83 eV best_site=4 E_ads_min=+0.598 eV elapsed=9.0s
```

(Positive `E_ads` means physisorption / repulsive — expected for CO₂ on pristine Cu(111) in the gas phase. Defective or doped Cu surfaces would be negative.)

## Pipeline 4 — Solid-state

**Scientific question.** How stiff is this crystal? The bulk modulus `B₀` tells you how much pressure changes the volume — it's a first-principles benchmark that's easy to compute and easy to compare against experiment.

**What happens in one task.**

1. Build a bulk structure via `ase.build.bulk(--structure)` (default `Si`), or load `--cif path.cif`.
2. Isotropically scale the unit cell across `--n-points` volumes spanning 90–110 % of equilibrium.
3. MACE-MP-0 single-point energy at each volume (no relaxation — that's the whole point of an EOS scan).
4. Fit the Energy(Volume) curve to the **Birch–Murnaghan** equation of state via `ase.eos.EquationOfState`.
5. Emit `task{N}_{structure}_eos.json` with V₀, E₀, B₀ (eV/Å³ and GPa) plus the raw (V, E) points.

**Key flags.** `--structure Si|Cu|NaCl|...`, `--cif path.cif`, `--n-points 20`, `--strain-low 0.90 --strain-high 1.10`, `--mace-model small|medium|large`.

**Smoke-test output.** Si, 9 points:
```
done V0=40.753 Å³ E0=-10.7407 eV B0=71.78 GPa elapsed=8.6s
```

(Experimental Si B₀ ≈ 97.9 GPa. MACE-MP-0 small/float32 underestimates; use `--mace-model large` and default-dtype float64 for better agreement.)

## Container quirks (shared across all pipelines)

- **AIMNet2 only.** Weights must be pre-staged and bind-mounted (the SIF is read-only). The launcher handles this for pharma (`--aimnet-assets` flag, default points at the shared staging dir).
- **MACE only.** Weights auto-download to `~/.cache/mace/` on first use. No pre-staging needed because `$HOME` is user-writable even inside a read-only SIF.
- **SSL cert env vars** (`SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `REQUESTS_CA_BUNDLE`) are injected into every `singularity exec` so any HTTPS call from inside the container works (HF model downloads, etc.).
- **Harmless startup noise:** `rm: cannot remove '/usr/local/cuda/compat/lib': Read-only file system` — ignore.
- **cuequivariance warning:** MACE falls back to standard e3nn without it. Slower but correct.

## Relationship to `tutorial/`

The `tutorial/` directory at the repo root is a different, older pipeline — a 4-phase high-throughput **molecular screening** workflow (GDB-17 → 3D embed → AIMNet2 relax → DFT HOMO/LUMO). It's complete and works end-to-end on Slurm.

The four pipelines here are orthogonal: each is a different scientific problem rather than another phase of the same screening pipeline. Think of them as four worked examples in a workshop rather than a sequence.

## Scaling up

- Bigger runs: drop `--dry-run` flags, raise `--n-tasks` / `--n-sites` / `--n-steps`, push `--concurrency` up when the partition is cold.
- Production runtimes on H100/H200: pharma ~1–2 min/molecule at 500 confs; battery ~1–2 min per 1000 MD steps; catalysis ~30 s/site; solid-state ~10 s/structure at 20 points.
- Out-of-memory: bump `--mem` in the launcher; `--mace-model` defaults to `small` — only move to `medium/large` when you need it.
