# THz pinch-harmonics — Meep reproduction

Meep/FDTD reproduction of the simulation in:

> H.-R. Park, Y.-M. Bahk, J. H. Choe, S. Han, S. S. Choi, K. J. Ahn,
> N. Park, Q.-H. Park, and D.-S. Kim, "Terahertz pinch harmonics enabled
> by single nano rods," *Opt. Express* **19**(24), 24775–24781 (2011).

## What the paper's FDTD model is

Section 2 ("To simulate pinch harmonics by Pt nano rods, we perform
three-dimensional finite-difference time-domain (FDTD) analysis using
Drude model...") describes:

- a single rectangular slot ("hole"), length **l1 = 10 μm**, width
  **w1 = 20 nm**, cut through a **20-nm-thick gold film** — a scaled-down
  stand-in for the real device's 300 μm × 120 nm slot antenna, chosen so
  the sub-skin-depth physics is preserved while the simulation stays
  tractable;
- a normally-incident, broadband THz pulse polarized along the slot's
  *short* axis (p-polarization, "perpendicular to the long axis of the
  rectangle");
- a single Pt nano rod sitting on top of the slot, oriented across it,
  with width **s = 60 nm** ("large") or **s = 20 nm** ("small") along the
  slot's long axis, placed either at the **1:2** position (Fig. 2) or the
  **1:1** midpoint (Fig. 3);
- reading out (a) the transmittance spectrum and (b) |Ex|² field maps
  just behind the exit face of the slot, at the fundamental and 3rd
  harmonic resonance.

This repo implements exactly that structure and those two measurements.
It does **not** attempt to reproduce the full-size experimental device
(300 μm slot, 120 nm width) — the paper itself only simulates the scaled
structure above.

## Files

| file | purpose |
|---|---|
| `materials.py` | metal media (see caveat below) |
| `geometry.py` | builds the slot + rod geometry as Meep objects |
| `simulate.py` | CLI driver: `--mode spectrum` (Fig. 2b/3b) or `--mode field` (Fig. 2c/3c) |
| `analyze.py` | loads simulate.py output, makes overlay/field plots, reports resonance peaks |
| `run_all.sh` | drives all the runs needed for Figs. 2 and 3 |
| `environment.yml` | conda-forge env spec (pymeep is not on PyPI) |

## Coordinate convention

`z` = propagation direction of the incident pulse; `x` = E-field
polarization = slot **width** direction (w1); `y` = slot **length**
direction (l1), i.e. the axis the rod's position ("1:2", "1:1") is
measured along. This matches the `Ex, Hy, kz` labels in Fig. 1(a) of the
paper.

## Honesty / caveats — please read before trusting numbers

This code was written and reviewed for **API correctness** against
Meep's documented Python interface, but **was not executed** in the
session that produced it: pymeep is a conda-forge-only package (its
PyPI listing is unrelated / fails to build) and installing a full
conda-forge pymeep stack plus running a meaningfully-resolved 3D FDTD job
was outside what was practical there. So:

1. **Not numerically validated.** No run of this code has actually been
   performed to confirm it reproduces the paper's curves. Please treat
   it as a carefully-written starting point, and sanity-check your first
   results (e.g. against the qualitative claims in the paper: a bare
   slot shows a dominant fundamental and weaker 3rd harmonic; a 600 nm
   rod at 1:2 splits/shifts the fundamental; a 250 nm rod at 1:2 makes
   the 3rd harmonic dominant; either rod at the 1:1 midpoint suppresses
   the fundamental, and the 20 nm-scale rod there additionally kills the
   symmetric n=2 mode).
2. **Metal model.** `materials.py` uses Meep's bundled
   `meep.materials.Au` / `.Pt` (a 6-pole Lorentz-Drude fit to Rakic et
   al., Appl. Opt. 37, 5271 (1998)), fitted against ~0.2–12 μm
   (near-IR–UV) optical data — *not* against THz/far-IR data. The paper
   itself uses a plain Drude model citing Ordal et al., Appl. Opt. 22,
   1099 (1983) (its Ref. [22]). In the simulated band the interband
   (Lorentzian) poles of the Rakic fit contribute negligibly and the
   free-electron Drude pole dominates, so this is a reasonable
   extrapolation and is *exactly* what ships with Meep (nothing
   hand-transcribed) — but it is still an extrapolation, and the two
   models' Drude parameters are not identical. For a quantitatively
   exact reproduction, look up Ordal's tabulated ω_p/ω_τ for Au and Pt
   yourself and pass them via `--custom-drude-film f_p,gamma` /
   `--custom-drude-rod f_p,gamma` (values in Meep frequency units, i.e.
   1/λ[μm], with the length unit a = 1 μm used throughout this project;
   1 cm⁻¹ = 1e-4 of that unit).
3. **Rod thickness.** The paper states the FDTD rod *widths* (60 nm /
   20 nm) but not its thickness for the scaled model. `--t2` defaults to
   20 nm (same as the film) as a reasonable placeholder — adjust it if
   you have a better basis for it.
4. **Rod length (l2).** Likewise unspecified for the scaled model;
   `--l2` defaults to 300 nm (must be > w1 = 20 nm so the rod actually
   bridges the slot, plus margin to sit on the film on both sides) —
   arbitrary and adjustable.
5. **Normalization.** Transmittance is computed as
   `P_trans(f; with structure) / P_inc(f; empty cell, same source and
   monitor-plane geometry)`, run as two independent Meep runs (`--empty`
   for the reference), which is the standard way to turn a flux spectrum
   into a transmittance spectrum and matches a THz-TDS sample/reference
   scan. It is not derived from the paper's own (unpublished) FDTD
   normalization procedure.
6. **Resolution / cost.** To resolve a 20 nm slot/rod with even a modest
   ~8–10 grid points needs `resolution` on the order of a few hundred
   (1/μm) over a domain spanning ~10+ μm — a genuinely large 3D job.
   `simulate.py`'s default (`resolution=60`) is a fast, low-fidelity
   *smoke test* to check the script runs and the geometry looks right,
   not a publication-quality setting. Expect to need MPI (`mpirun -np
   N`) and a machine with real memory/core budget for a faithful run;
   consider Meep's chunk/load-balancing options if scaling to many
   ranks.
7. **PML thickness.** The source pulse's bandwidth (default `--fcen
   0.09 --df 0.16`, i.e. wavelengths roughly 3–300 μm in this unit
   system) is very broad relative to the sub-μm structure, so the PML
   (`--dpml`, default 1 μm) will absorb the low-frequency tail somewhat
   imperfectly. If you see reference-run transmittance deviating
   noticeably from 1 outside the frequency band you actually care about,
   narrow `--fcen/--df` to your region of interest and/or increase
   `--dpml`.

## Installing pymeep

```
conda env create -f environment.yml
conda activate pinch-harmonics
```

(Or `conda install -c conda-forge pymeep=*=mpi_mpich_*` into an existing
environment. See the Meep project's own installation docs for
alternatives such as a serial build.)

## Running

```
bash run_all.sh                 # fast smoke test at resolution=60
RESOLUTION=300 NP=32 bash run_all.sh   # a much more serious run
```

or drive `simulate.py` directly — see `python3 simulate.py --help` and
the module docstring at the top of the file for example invocations of
`--mode spectrum` and `--mode field`.
