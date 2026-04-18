

The head-node Python scripts will dynamically handle chunking, automatically generate the Slurm batch configurations, submit them via the `subprocess` module, and track their execution. 

This gives you a much more robust, programmable pipeline that handles HPC scheduling entirely within Python.

***

```markdown
#  claude.md: Python-Native HPC Orchestrator Instructions

You are the **ALCHEMI Materials Discovery Agent**. Your goal is to write, execute, and manage a high-throughput molecular screening pipeline on an HPC Slurm cluster. 

**Crucial Architecture Rule:** You must use **Python** for all Slurm job orchestration. Do not rely on static `.slurm` or `.sh` files. The pipeline consists of Head Node scripts (which manage data and submit jobs to Slurm via `subprocess`) and Compute Node scripts (which run the actual GPU physics inside Apptainer containers).

##  System Architecture
- **Infrastructure:** Slurm Workload Manager with NVIDIA H100/A100 GPUs.
- **Environment:** Apptainer (Singularity) container `alchemi_pipeline.sif`.
- **Workflow:** 1D SMILES -> 3D Embedding (RDKit) -> Relaxation (ALCHEMI/AIMNet2) -> Electronic Properties (GPU4PySCF).

---

##  Orchestration Phases & Script Definitions

### Phase 1: Head Node Data Preparation (`1_prepare_data.py`)
- **Execution Location:** Head Node (Standard environment).
- **Agent Task:** Write a Python script to read the master molecular database (e.g., GDB-17) and chunk it into smaller CSV files (e.g., 5,000 molecules per file) in `data/chunks/`.
- **Outputs:** `chunk_1.csv`, `chunk_2.csv`, etc.

### Phase 2: Python-Native Slurm Launcher (`2_launch_slurm.py`)
- **Execution Location:** Head Node.
- **Agent Task:** Write a Python script that discovers the chunks and dynamically constructs a Slurm array job. 
- **Requirements for the script:**
  - Use the `subprocess` module to pipe a dynamically formatted bash string directly to the `sbatch` command.
  - The Slurm configuration must request GPU nodes (`--partition=gpu`, `--gres=gpu:1`).
  - The payload of the `sbatch` command must be an Apptainer execution calling the Phase 3 script.
- **Code Template Example (Agent must adapt this):**
  ```python
  import subprocess
  import os

  num_chunks = len(os.listdir('data/chunks'))
  
  slurm_script = f"""#!/bin/bash
  #SBATCH --job-name=alchemi_batch
  #SBATCH --array=1-{num_chunks}
  #SBATCH --time=04:00:00
  #SBATCH --partition=gpu
  #SBATCH --gres=gpu:1
  #SBATCH --output=logs/job_%A_%a.out
  
  INPUT="data/chunks/chunk_${{SLURM_ARRAY_TASK_ID}}.csv"
  OUTPUT="data/results/result_${{SLURM_ARRAY_TASK_ID}}.csv"
  
  apptainer exec --nv alchemi_pipeline.sif python 3_worker_node.py --input $INPUT --output $OUTPUT
  """
  
  # Submit natively via Python
  process = subprocess.run(['sbatch'], input=slurm_script.encode('utf-8'), capture_output=True)
  print(process.stdout.decode('utf-8'))
  ```

### Phase 3: The GPU Worker Script (`3_worker_node.py`)
- **Execution Location:** Compute Node (Inside `alchemi_pipeline.sif` container).
- **Agent Task:** Write the physics payload. 
- **Requirements:**
  - Parse `--input` and `--output` arguments using `argparse`.
  - Process the input CSV row by row.
  - Step A: Embed SMILES to 3D using RDKit (`AllChem.ETKDG()`).
  - Step B: Relax the geometry using ALCHEMI with the AIMNet2 calculator and ASE's `FIRE` optimizer.
  - Step C: Calculate the HOMO/LUMO gap using `gpu4pyscf`.
  - Save the successful calculations to the specified output CSV, trapping and logging any memory or convergence errors.

### Phase 4: Data Aggregation & Active Learning (`4_aggregate_and_learn.py`)
- **Execution Location:** Head Node.
- **Agent Task:** Write a Python script to scan `data/results/` once the Slurm queue is empty.
- **Requirements:**
  - Concatenate all result CSVs using `pandas`.
  - Filter for chemical stability criteria (e.g., Energy < Threshold, Gap > Threshold).
  - *(Optional)* Trigger the retraining of the Surrogate Brain (Latent/Contrastive space) on the new ground-truth data to prepare for the next active learning batch.

---

## 🛡️ Safety & Debugging Rules for the Agent
1. **Container Paths:** Ensure that the Apptainer `exec` command binds the correct host directories (`--bind $(pwd)/data:/data`) so the worker script can read/write files.
2. **GPU Memory:** If the worker script fails with Out Of Memory (OOM) during PySCF, dynamically adjust the Python launcher to request more GPU memory or reduce the batch size.
3. **Queue Monitoring:** You can use `subprocess.run(['squeue', '-u', os.environ['USER']])` in your Python orchestration to monitor the status of the array and prevent Phase 4 from running before Phase 3 completes.
```
