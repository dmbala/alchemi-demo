"""Battery MD: Li+ in an electrolyte box, Langevin NVT with MACE-MP-0, Li-O RDF.

Runs inside alchemi_MACE.sif on a GPU node. One Slurm task processes one system.
"""

import argparse
import json
import logging
import pathlib
import sys
import time


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Battery MD + RDF worker.")
    p.add_argument("--task-id", type=int, default=1)
    p.add_argument("--initial-box", type=pathlib.Path, default=None,
                   help="Optional extxyz/xyz of initial system. If omitted, a small"
                        " 10-H2O + 1-Li+ cubic box is generated.")
    p.add_argument("--output-dir", required=True, type=pathlib.Path)
    p.add_argument("--n-steps", type=int, default=1000,
                   help="Langevin NVT steps (spec target: 10000).")
    p.add_argument("--timestep-fs", type=float, default=1.0)
    p.add_argument("--temperature-k", type=float, default=300.0)
    p.add_argument("--friction-inv-fs", type=float, default=0.01)
    p.add_argument("--sample-every", type=int, default=10,
                   help="Record positions every N steps for the RDF histogram.")
    p.add_argument("--rdf-rmax", type=float, default=6.0)
    p.add_argument("--rdf-nbins", type=int, default=60)
    p.add_argument("--mace-model", default="small",
                   help="MACE-MP-0 size: 'small' | 'medium' | 'large'")
    return p.parse_args()


def build_default_box(n_waters: int = 10, box_angstrom: float = 12.0, seed: int = 0):
    """Pack n_waters H2O + 1 Li+ into a cubic PBC box via random non-overlapping insertion."""
    import numpy as np
    from ase import Atoms
    from ase.build import molecule

    rng = np.random.default_rng(seed)
    placed = Atoms(cell=[box_angstrom] * 3, pbc=True)

    def min_distance(trial, existing) -> float:
        if len(existing) == 0:
            return float("inf")
        dx = trial.get_positions()[:, None, :] - existing.get_positions()[None, :, :]
        dx -= np.round(dx / box_angstrom) * box_angstrom
        return float(np.linalg.norm(dx, axis=-1).min())

    for _ in range(n_waters):
        for _ in range(200):
            w = molecule("H2O")
            w.rotate(360 * rng.random(), "x")
            w.rotate(360 * rng.random(), "y")
            w.rotate(360 * rng.random(), "z")
            w.translate(rng.uniform(0, box_angstrom, 3))
            if min_distance(w, placed) > 2.2:
                placed += w
                break
        else:
            raise RuntimeError("could not pack water (box too small)")

    li = Atoms("Li", positions=[[box_angstrom / 2] * 3])
    placed += li
    return placed


def load_or_build_box(args):
    if args.initial_box is not None:
        from ase.io import read
        return read(str(args.initial_box))
    return build_default_box()


def compute_li_o_rdf(positions_traj, cell, r_max: float, n_bins: int,
                     li_idx, o_idx):
    """Histogram Li–O pair distances across all recorded frames. Returns (r, g(r))."""
    import numpy as np

    box = np.diag(cell) if cell.ndim == 2 else cell
    volume = float(np.prod(box))
    edges = np.linspace(0.0, r_max, n_bins + 1)
    shell_volumes = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    hist = np.zeros(n_bins)
    n_frames = len(positions_traj)
    n_o = len(o_idx)

    for pos in positions_traj:
        for i in li_idx:
            d = pos[o_idx] - pos[i]
            d -= np.round(d / box) * box
            r = np.linalg.norm(d, axis=1)
            r = r[r < r_max]
            hist += np.histogram(r, bins=edges)[0]

    density_o = n_o / volume
    g = hist / (n_frames * len(li_idx) * shell_volumes * density_o)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, g


def main() -> int:
    args = parse_args()
    logging.basicConfig(level="INFO", format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    from ase.io import write as ase_write
    from ase.md.langevin import Langevin
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from ase import units
    from mace.calculators import mace_mp
    import numpy as np

    atoms = load_or_build_box(args)
    symbols = np.array(atoms.get_chemical_symbols())
    li_idx = np.where(symbols == "Li")[0]
    o_idx = np.where(symbols == "O")[0]
    logging.info("system size: n_atoms=%d n_Li=%d n_O=%d", len(atoms), len(li_idx), len(o_idx))

    atoms.calc = mace_mp(model=args.mace_model, default_dtype="float32", device="cuda")

    MaxwellBoltzmannDistribution(atoms, temperature_K=args.temperature_k)
    dyn = Langevin(
        atoms,
        timestep=args.timestep_fs * units.fs,
        temperature_K=args.temperature_k,
        friction=args.friction_inv_fs / units.fs,
    )

    trajectory = []

    def snapshot():
        trajectory.append(atoms.get_positions().copy())

    dyn.attach(snapshot, interval=args.sample_every)

    t0 = time.time()
    dyn.run(args.n_steps)
    md_elapsed = time.time() - t0
    logging.info("MD done: %d steps in %.1fs (%d snapshots)",
                 args.n_steps, md_elapsed, len(trajectory))

    r_centers, g_r = compute_li_o_rdf(
        trajectory, np.array(atoms.get_cell()), args.rdf_rmax, args.rdf_nbins, li_idx, o_idx,
    )

    prefix = f"task{args.task_id}"
    (args.output_dir / f"{prefix}_rdf_li_o.json").write_text(json.dumps({
        "r_angstrom": r_centers.tolist(),
        "g_r": g_r.tolist(),
        "n_frames": len(trajectory),
        "n_steps": args.n_steps,
        "temperature_K": args.temperature_k,
    }))
    ase_write(str(args.output_dir / f"{prefix}_final_frame.xyz"), atoms)
    logging.info("wrote RDF + final frame -> %s/", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
