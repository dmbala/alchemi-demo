"""Solid-state EOS: strain the unit cell 90-110%, single-point MACE, fit Birch-Murnaghan.

Builds a bulk crystal from either `--cif` or `--structure {Si,Cu,NaCl,...}` (via ase.build.bulk).
"""

import argparse
import json
import logging
import pathlib
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Solid-state EOS worker.")
    p.add_argument("--task-id", type=int, default=1)
    p.add_argument("--cif", type=pathlib.Path, default=None,
                   help="Optional CIF path. If omitted, a bulk structure is generated.")
    p.add_argument("--structure", default="Si",
                   help="Element or compound name passed to ase.build.bulk (e.g., Si, Cu, NaCl).")
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--n-points", type=int, default=20,
                   help="Number of volume points spanning 90-110% of V0.")
    p.add_argument("--strain-low", type=float, default=0.90)
    p.add_argument("--strain-high", type=float, default=1.10)
    p.add_argument("--mace-model", default="small")
    return p.parse_args()


# ASE's bulk() has reference data only for elemental solids; compounds need
# crystalstructure + lattice constant. Small recipe book for common tutorial cases.
_COMPOUND_RECIPES = {
    "NaCl": {"crystalstructure": "rocksalt", "a": 5.64},
    "MgO":  {"crystalstructure": "rocksalt", "a": 4.21},
    "GaAs": {"crystalstructure": "zincblende", "a": 5.65},
    "LiF":  {"crystalstructure": "rocksalt", "a": 4.03},
}


def build_system(args):
    from ase.build import bulk
    from ase.io import read
    if args.cif is not None:
        return read(str(args.cif))
    if args.structure in _COMPOUND_RECIPES:
        return bulk(args.structure, **_COMPOUND_RECIPES[args.structure])
    return bulk(args.structure)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    from ase.eos import EquationOfState
    from ase.io import write as ase_write
    from mace.calculators import mace_mp

    atoms = build_system(args)
    if atoms.cell.rank != 3:
        raise ValueError("EOS requires a 3D periodic cell")
    calc = mace_mp(model=args.mace_model, default_dtype="float32", device="cuda")
    atoms.calc = calc

    v0_factor = np.linspace(args.strain_low, args.strain_high, args.n_points) ** (1.0 / 3.0)
    original_cell = atoms.get_cell().copy()
    original_positions = atoms.get_scaled_positions()

    volumes, energies = [], []
    t0 = time.time()
    for f in v0_factor:
        strained = atoms.copy()
        strained.set_cell(original_cell * f, scale_atoms=False)
        strained.set_scaled_positions(original_positions)
        strained.calc = calc
        volumes.append(float(strained.get_volume()))
        energies.append(float(strained.get_potential_energy()))

    from ase import units
    eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
    v0, e0, b0 = eos.fit()
    b0_gpa = b0 / units.kJ * 1.0e24

    out = {
        "structure": args.structure if args.cif is None else str(args.cif),
        "n_points": len(volumes),
        "volumes_A3": volumes,
        "energies_eV": energies,
        "V0_A3": float(v0),
        "E0_eV": float(e0),
        "B0_eV_per_A3": float(b0),
        "B0_GPa": float(b0_gpa),
    }
    prefix = f"task{args.task_id}_{args.structure if args.cif is None else args.cif.stem}"
    (args.output_dir / f"{prefix}_eos.json").write_text(json.dumps(out, indent=2))
    ase_write(str(args.output_dir / f"{prefix}_relaxed.xyz"), atoms)

    logging.info("done V0=%.3f Å³ E0=%.4f eV B0=%.2f GPa elapsed=%.1fs",
                 v0, e0, b0_gpa, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
