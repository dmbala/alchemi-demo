"""GPU worker: embed, relax, compute HOMO/LUMO. Runs inside alchemi_ht.sif."""

import argparse
import csv
import logging
import pathlib
import sys
import time
import traceback

HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ALCHEMI HT worker.")
    p.add_argument("--input", required=True, type=pathlib.Path)
    p.add_argument("--output", required=True, type=pathlib.Path)
    p.add_argument("--xc", default="B3LYP")
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-opt-steps", type=int, default=200)
    p.add_argument("--aimnet-model", default="/aimnet_assets/aimnet2_wb97m_d3_0.pt",
                   help="Absolute path inside container to a pre-staged AIMNet2 .pt file.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def embed_3d(smiles: str):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 0xC0FFEE
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise RuntimeError("ETKDGv3 embedding failed")
    AllChem.MMFFOptimizeMolecule(mol, maxIters=200)
    return mol


def rdkit_to_ase(mol):
    from ase import Atoms

    conf = mol.GetConformer()
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = conf.GetPositions()
    return Atoms(symbols=symbols, positions=positions)


def make_aimnet2_calculator(model_path: str):
    from aimnet.calculators import AIMNet2ASE
    from aimnet.calculators.calculator import AIMNet2Calculator

    base = AIMNet2Calculator(model_path, needs_dispersion=False)
    return AIMNet2ASE(base)


def relax(atoms, calc, fmax: float, max_steps: int):
    from ase.optimize import FIRE

    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=max_steps)
    return atoms


def compute_homo_lumo(atoms, xc: str, basis: str):
    from pyscf import gto
    from gpu4pyscf import dft

    atom_spec = [
        (a.symbol, (float(a.position[0]), float(a.position[1]), float(a.position[2])))
        for a in atoms
    ]
    mol = gto.M(atom=atom_spec, basis=basis, charge=0, spin=0, verbose=0)
    mf = dft.RKS(mol, xc=xc)
    energy = float(mf.kernel())

    occupied = mf.mo_energy[mf.mo_occ > 0]
    virtual = mf.mo_energy[mf.mo_occ == 0]
    if len(occupied) == 0 or len(virtual) == 0:
        raise RuntimeError("no HOMO/LUMO pair found")
    return energy, float(occupied[-1]) * HARTREE_TO_EV, float(virtual[0]) * HARTREE_TO_EV


def process_row(smiles: str, calc, args) -> dict:
    mol = embed_3d(smiles)
    atoms = rdkit_to_ase(mol)
    atoms = relax(atoms, calc, args.fmax, args.max_opt_steps)
    energy, homo_ev, lumo_ev = compute_homo_lumo(atoms, args.xc, args.basis)
    return {
        "smiles": smiles,
        "energy": energy,
        "homo": homo_ev,
        "lumo": lumo_ev,
        "gap": lumo_ev - homo_ev,
        "n_atoms": len(atoms),
    }


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Heavy: loads the .pt model + allocates GPU tensors. Once per process, not per row.
    calc = make_aimnet2_calculator(args.aimnet_model)

    import torch

    fields = ["smiles", "energy", "homo", "lumo", "gap", "n_atoms", "source_id"]
    n_ok = n_fail = 0
    t0 = time.time()

    with args.input.open() as fin, args.output.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()

        for idx, row in enumerate(reader):
            smiles = row["smiles"]
            try:
                result = process_row(smiles, calc, args)
                result["source_id"] = row.get("source_id", "")
                writer.writerow(result)
                n_ok += 1
            except Exception as exc:
                n_fail += 1
                logging.warning(
                    "row=%d smiles=%s FAILED: %s\n%s",
                    idx, smiles, exc, traceback.format_exc(),
                )
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    logging.info("done n_ok=%d n_fail=%d elapsed=%.1fs", n_ok, n_fail, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
