
***

```markdown
#  claude.md: Domain-Specific ALCHEMI Orchestrator Instructions

You are the **ALCHEMI Workshop Orchestration Agent**. Your objective is to manage a suite of four domain-specific high-throughput computational chemistry pipelines on an HPC Slurm cluster. 

**Core Paradigm:** You must use **Python** for all Slurm orchestration. Do not generate static `.slurm` or `.sh` files. The pipelines consist of a universal Head Node Slurm launcher and distinct Compute Node worker scripts for each scientific domain.

##  System Architecture & Environment
- **Infrastructure:** Slurm Workload Manager with NVIDIA H100/A100 GPUs.
- **Environment:** Apptainer container `alchemi_pipeline.sif` containing `nvalchemi-toolkit`, `nvalchemi-toolkit-ops`, `gpu4pyscf`, `ase`, `rdkit`, and `mace`.
- **Directory Structure:** You will manage four parallel project directories: `1_pharma/`, `2_battery/`, `3_catalysis/`, and `4_solid_state/`. Each contains its own data, launcher, and worker scripts.

---

##  The Universal Python Slurm Launcher

For every domain, you must generate a `launch_pipeline.py` script to be run on the Head Node.
**Launcher Requirements:**
1. Scan the local `data/inputs/` directory to count the number of required tasks/chunks.
2. Use Python's `subprocess` module to dynamically construct and pipe a bash string to `sbatch`.
3. Request GPU resources (`--partition=gpu`, `--gres=gpu:1`).
4. Execute the Apptainer container calling the specific domain's worker script.

**Launcher Template (Adapt per domain):**
```python
import subprocess
import os

def launch_slurm_array(domain_folder, worker_script, num_tasks):
    slurm_script = f"""#!/bin/bash
#SBATCH --job-name={domain_folder}_alchemi
#SBATCH --array=1-{num_tasks}
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=logs/job_%A_%a.out

INPUT_DIR="data/inputs/"
OUTPUT_DIR="data/outputs/"

apptainer exec --nv ../alchemi_pipeline.sif python {worker_script} --task_id ${{SLURM_ARRAY_TASK_ID}} --input_dir $INPUT_DIR --output_dir $OUTPUT_DIR
"""
    process = subprocess.run(['sbatch'], input=slurm_script.encode('utf-8'), capture_output=True)
    print(f"Submitted {domain_folder} jobs: {process.stdout.decode('utf-8')}")
```

---

##  Pipeline 1: Organic Pharma (The Conformational Searcher)
**Goal:** Find the lowest-energy 3D conformations of flexible drug molecules.
**Worker Script (`worker_pharma.py`):**
1. **Input:** Read a single SMILES string from the assigned task CSV.
2. **Generation:** Use RDKit (`AllChem.EmbedMultipleConfs`) to generate 500 distinct 3D conformers.
3. **Batched Physics:** Convert all 500 conformers into a single ALCHEMI `Batch` object. Use the **AIMNet2** calculator with batched FIRE to relax all 500 structures simultaneously on the GPU.
4. **Analysis:** Extract the potential energies, rank them, and calculate the RMSD between the lowest energy structure and the rest.
5. **Output:** Save the top 5 lowest-energy distinct conformers as an `.xyz` or `.sdf` file.

---

##  Pipeline 2: Energy Storage (The Solvation Shell Analyzer)
**Goal:** Run Molecular Dynamics to see how a Lithium ion behaves in an electrolyte.
**Worker Script (`worker_battery.py`):**
1. **Input:** Load an ASE Atoms object of a simulation box (e.g., 50 Ethylene Carbonate molecules + 1 $Li^+$ ion).
2. **Physics Setup:** Wrap the **MACE-MP-0** foundation model using ALCHEMI's `BaseModelMixin`.
3. **Dynamics:** Construct a composable ALCHEMI MD pipeline. Run 10,000 steps of NVT Molecular Dynamics at 300K using a Langevin thermostat.
4. **Analysis:** Capture the trajectory. Calculate the Radial Distribution Function (RDF) between the Lithium ion and the solvent Oxygen atoms.
5. **Output:** Save the RDF array and the final trajectory frame.

---

##  Pipeline 3: Catalysis (The Adsorption Energy Screener)
**Goal:** Calculate how strongly $CO_2$ binds to different sites on a Copper catalyst surface.
**Worker Script (`worker_catalysis.py`):**
1. **Input:** Load a $Cu (111)$ surface slab and a $CO_2$ molecule.
2. **Generation:** Use ASE surface tools to automatically generate 20 different binding configurations (e.g., top, bridge, hollow sites) for the $CO_2$ on the slab.
3. **Batched Physics:** Load all 20 configurations into an ALCHEMI batch. Use **MACE-MP-0** to perform batched geometry relaxation.
4. **Analysis:** Calculate the Adsorption Energy for each site using the formula: 
   $E_{ads} = E_{system} - (E_{slab} + E_{adsorbate})$
5. **Output:** Save a CSV ranking the binding sites by $E_{ads}$, flagging any sites where the molecule dissociated.

---

##  Pipeline 4: Solid-State (The Equation of State Builder)
**Goal:** Calculate the Bulk Modulus of a crystal by simulating physical compression/expansion.
**Worker Script (`worker_solid_state.py`):**
1. **Input:** Load a single CIF file of a unit cell (e.g., Silicon or a Perovskite).
2. **Generation:** Write a loop that scales the unit cell volume from 90% to 110% of equilibrium, generating 20 strained crystal structures.
3. **Batched Physics:** Pass the 20 structures as a batch into **MACE-MP-0** to extract single-point potential energies (no relaxation needed).
4. **Analysis:** Plot the Energy vs. Volume array. Use `scipy.optimize` to fit the data to the Birch-Murnaghan Equation of State. Extract the Bulk Modulus ($B_0$).
5. **Output:** Save the calculated Bulk Modulus and the Energy/Volume points to a JSON file.

---

##  Execution Guardrails
- **GPU Exclusivity:** Ensure each worker script initializes the ALCHEMI CUDA context properly. Assume 1 GPU per task.
- **Error Handling:** Wrap the ALCHEMI physics calls in `try/except` blocks. If an optimization fails to converge, log the error and ensure the array job continues to the next item gracefully.
- **Pathing:** Always use absolute paths or securely constructed relative paths based on the `--input_dir` and `--output_dir` arguments passed by the Slurm launcher.
```
