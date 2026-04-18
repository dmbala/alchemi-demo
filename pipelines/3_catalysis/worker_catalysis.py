"""Catalysis adsorption screen: Cu(111) + CO2 at N binding sites, MACE relax, E_ads.

Generates the slab and sites deterministically from ASE. One Slurm task does the full
screen for one system (reference calculations + N combined relaxations).
"""

import argparse
import csv
import logging
import pathlib
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Catalysis adsorption worker.")
    p.add_argument("--task-id", type=int, default=1)
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--metal", default="Cu")
    p.add_argument("--facet-size", nargs=2, type=int, default=(3, 3),
                   help="In-plane lattice repetitions (a, b).")
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--vacuum-angstrom", type=float, default=10.0)
    p.add_argument("--n-sites", type=int, default=20,
                   help="Binding configurations to generate (grid + small random jitter).")
    p.add_argument("--initial-height", type=float, default=2.0,
                   help="Height (Å) of CO2 above the topmost Cu layer.")
    p.add_argument("--fmax", type=float, default=0.05)
    p.add_argument("--max-opt-steps", type=int, default=100)
    p.add_argument("--mace-model", default="small")
    return p.parse_args()


def build_slab(metal, size, n_layers, vacuum):
    from ase.build import fcc111
    slab = fcc111(metal, size=(size[0], size[1], n_layers), vacuum=vacuum)
    slab.pbc = (True, True, True)
    return slab


def generate_sites(slab, n_sites, initial_height, seed: int = 0):
    """Distribute N CO2 positions across the slab's xy area with small random jitter."""
    import numpy as np
    from ase.build import molecule

    rng = np.random.default_rng(seed)
    cell = slab.get_cell()
    top_z = max(slab.get_positions()[:, 2])

    n_grid = int(np.ceil(np.sqrt(n_sites)))
    grid_xy = np.linspace(0.05, 0.95, n_grid)
    positions = []
    for i in grid_xy:
        for j in grid_xy:
            xy = i * cell[0, :2] + j * cell[1, :2]
            jitter = rng.uniform(-0.3, 0.3, 2)
            positions.append((xy[0] + jitter[0], xy[1] + jitter[1]))
            if len(positions) == n_sites:
                break
        if len(positions) == n_sites:
            break

    systems = []
    for idx, (x, y) in enumerate(positions):
        co2 = molecule("CO2")
        co2.translate((x, y, top_z + initial_height))
        sys_atoms = slab.copy() + co2
        systems.append((idx + 1, sys_atoms))
    return systems


def fix_bottom_layers(atoms, n_free_layers: int = 2):
    """Freeze every layer except the top n_free_layers + the adsorbate."""
    import numpy as np
    from ase.constraints import FixAtoms

    z = atoms.get_positions()[:, 2]
    # Adsorbate atoms are C and O; everything else is the metal. Keep adsorbate free.
    metal_mask = np.array([s != "C" and s != "O" for s in atoms.get_chemical_symbols()])
    metal_z = z[metal_mask]
    if len(metal_z) == 0:
        return
    sorted_z = sorted(set(np.round(metal_z, 2)))
    free_zs = set(sorted_z[-n_free_layers:])
    fix_indices = [
        i for i in range(len(atoms))
        if metal_mask[i] and round(z[i], 2) not in free_zs
    ]
    atoms.set_constraint(FixAtoms(indices=fix_indices))


def relax(atoms, calc, fmax, max_steps):
    from ase.optimize import FIRE
    atoms.calc = calc
    FIRE(atoms, logfile=None).run(fmax=fmax, steps=max_steps)
    return float(atoms.get_potential_energy())


def main() -> int:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ase.build import molecule
    from ase.io import write as ase_write
    from mace.calculators import mace_mp

    calc = mace_mp(model=args.mace_model, default_dtype="float32", device="cuda")
    t0 = time.time()

    logging.info("reference: clean slab")
    slab = build_slab(args.metal, args.facet_size, args.n_layers, args.vacuum_angstrom)
    fix_bottom_layers(slab, n_free_layers=2)
    e_slab = relax(slab, calc, args.fmax, args.max_opt_steps)

    logging.info("reference: isolated CO2")
    co2 = molecule("CO2")
    co2.set_cell([20.0, 20.0, 20.0])
    co2.center()
    co2.pbc = (True, True, True)
    co2.calc = calc
    from ase.optimize import FIRE
    FIRE(co2, logfile=None).run(fmax=args.fmax, steps=args.max_opt_steps)
    e_co2 = float(co2.get_potential_energy())

    logging.info("generating %d sites and relaxing", args.n_sites)
    systems = generate_sites(slab, args.n_sites, args.initial_height)

    results = []
    for site_id, sys_atoms in systems:
        fix_bottom_layers(sys_atoms, n_free_layers=2)
        try:
            e_system = relax(sys_atoms, calc, args.fmax, args.max_opt_steps)
            e_ads = e_system - (e_slab + e_co2)
            # Dissociation check: C–O bond length blow-up.
            pos = sys_atoms.get_positions()
            c_idx = [i for i, s in enumerate(sys_atoms.get_chemical_symbols()) if s == "C"]
            o_idx = [i for i, s in enumerate(sys_atoms.get_chemical_symbols()) if s == "O"]
            if c_idx and o_idx:
                import numpy as np
                max_co_bond = float(np.linalg.norm(pos[c_idx[0]] - pos[o_idx[-2:]], axis=1).max())
            else:
                max_co_bond = float("nan")
            dissociated = max_co_bond > 2.0
            results.append({
                "site": site_id, "E_system_eV": e_system, "E_ads_eV": e_ads,
                "max_CO_bond_A": max_co_bond, "dissociated": dissociated,
            })
            ase_write(str(args.output_dir / f"task{args.task_id}_site{site_id}.xyz"), sys_atoms)
        except Exception as exc:
            logging.warning("site %d failed: %s", site_id, exc)
            results.append({
                "site": site_id, "E_system_eV": float("nan"),
                "E_ads_eV": float("nan"), "max_CO_bond_A": float("nan"),
                "dissociated": False, "error": str(exc)[:80],
            })

    results.sort(key=lambda r: (r["E_ads_eV"] if r["E_ads_eV"] == r["E_ads_eV"] else float("inf")))
    csv_path = args.output_dir / f"task{args.task_id}_adsorption.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sorted({k for r in results for k in r}))
        w.writeheader()
        w.writerows(results)

    logging.info("done E_slab=%.4f eV E_CO2=%.4f eV best_site=%d E_ads_min=%.3f eV elapsed=%.1fs",
                 e_slab, e_co2, results[0]["site"], results[0]["E_ads_eV"], time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
