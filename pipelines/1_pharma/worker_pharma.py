"""Pharma conformer search: SMILES -> 500 ETKDG conformers -> AIMNet2 relax -> top-5 xyz.

Runs inside alchemi_MACE.sif on a GPU node. Called by launch_pipeline.py with a task id
that indexes into the input CSV.
"""

import argparse
import csv
import logging
import pathlib
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pharma worker.")
    p.add_argument("--input", required=True, type=pathlib.Path,
                   help="CSV with columns: name,smiles")
    p.add_argument("--task-id", type=int, required=True,
                   help="1-indexed row to process (Slurm array task id).")
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--aimnet-model", default="/aimnet_assets/aimnet2_wb97m_d3_0.pt")
    p.add_argument("--n-confs", type=int, default=500)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-opt-steps", type=int, default=200)
    p.add_argument("--rmsd-threshold", type=float, default=0.5,
                   help="Discard conformers within this RMSD (Å) of a higher-ranked one.")
    return p.parse_args()


def read_molecule(input_csv: pathlib.Path, task_id: int):
    with input_csv.open() as f:
        rows = list(csv.DictReader(f))
    if task_id < 1 or task_id > len(rows):
        raise IndexError(f"task_id {task_id} out of range [1, {len(rows)}]")
    return rows[task_id - 1]


def generate_conformers(smiles: str, n_confs: int):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    params.numThreads = 0
    conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(conf_ids) == 0:
        raise RuntimeError("no conformers embedded")
    AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=0)
    return mol, list(conf_ids)


def mol_conf_to_atoms(mol, conf_id):
    from ase import Atoms
    conf = mol.GetConformer(conf_id)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    positions = conf.GetPositions()
    return Atoms(symbols=symbols, positions=positions)


def relax_all(mol, conf_ids, calc, fmax: float, max_steps: int):
    """Relax each conformer independently with AIMNet2. Returns list of (ase.Atoms, energy_eV)."""
    from ase.optimize import FIRE

    results = []
    for cid in conf_ids:
        atoms = mol_conf_to_atoms(mol, cid)
        atoms.calc = calc
        try:
            FIRE(atoms, logfile=None).run(fmax=fmax, steps=max_steps)
            results.append((atoms, float(atoms.get_potential_energy())))
        except Exception as exc:
            logging.warning("conformer %d relax failed: %s", cid, exc)
    return results


def rmsd(a, b) -> float:
    """Minimum-image-agnostic atom-order-preserving RMSD after Kabsch alignment."""
    import numpy as np

    pa = a.get_positions() - a.get_positions().mean(axis=0)
    pb = b.get_positions() - b.get_positions().mean(axis=0)
    h = pa.T @ pb
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    r = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return float(np.sqrt(np.mean(np.sum((pa @ r.T - pb) ** 2, axis=1))))


def select_diverse_top(relaxed, top_k: int, rmsd_thresh: float):
    """Pick top-k by energy, dropping any within `rmsd_thresh` of a higher-ranked pick."""
    ordered = sorted(relaxed, key=lambda t: t[1])
    keep = []
    for atoms, e in ordered:
        if len(keep) == top_k:
            break
        if all(rmsd(atoms, prev) > rmsd_thresh for prev, _ in keep):
            keep.append((atoms, e))
    return keep


def write_xyz(path: pathlib.Path, selected, name: str) -> None:
    from ase.io import write

    atoms_list = []
    for i, (atoms, e) in enumerate(selected):
        atoms.info = {"name": name, "rank": i + 1, "energy_eV": e}
        atoms_list.append(atoms)
    write(str(path), atoms_list, format="extxyz")


def main() -> int:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    row = read_molecule(args.input, args.task_id)
    name, smiles = row["name"], row["smiles"]
    logging.info("task=%d name=%s smiles=%s", args.task_id, name, smiles)

    from aimnet.calculators import AIMNet2ASE
    from aimnet.calculators.calculator import AIMNet2Calculator

    calc = AIMNet2ASE(AIMNet2Calculator(args.aimnet_model, needs_dispersion=False))

    t0 = time.time()
    mol, conf_ids = generate_conformers(smiles, args.n_confs)
    logging.info("embedded %d conformers", len(conf_ids))

    relaxed = relax_all(mol, conf_ids, calc, args.fmax, args.max_opt_steps)
    logging.info("relaxed %d/%d conformers", len(relaxed), len(conf_ids))

    selected = select_diverse_top(relaxed, args.top_k, args.rmsd_threshold)
    out_xyz = args.output_dir / f"{name}_top{len(selected)}.xyz"
    write_xyz(out_xyz, selected, name)

    logging.info(
        "done top=%d min_energy=%.4f eV elapsed=%.1fs -> %s",
        len(selected), selected[0][1] if selected else float("nan"), time.time() - t0, out_xyz,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
