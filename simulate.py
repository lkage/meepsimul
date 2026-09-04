#!/usr/bin/env python3
"""3D Meep FDTD reproduction of the single-nano-rod THz "pinch harmonics"
nanoresonator from:

  H.-R. Park et al., "Terahertz pinch harmonics enabled by single nano
  rods," Opt. Express 19(24), 24775-24781 (2011).

This script reproduces the FDTD part of the paper (Sec. 2, paragraph
starting "To simulate pinch harmonics by Pt nano rods..."), i.e. a single
finite rectangular slot (length l1 = 10 um, width w1 = 20 nm) in a 20-nm
thick gold film, illuminated at normal incidence by a p-polarized THz
pulse (E along the slot's short axis), with an optional Pt nano rod
(width s = 60 nm "large" or 20 nm "small") placed on top of the slot at
either the 1:2 position (Fig. 2) or the 1:1 midpoint (Fig. 3).

Two modes:
  spectrum  -- broadband flux run -> transmittance(f) = P_trans(f)/P_inc(f)
               (reproduces Fig. 2(b) / Fig. 3(b))
  field     -- narrowband DFT-field run at explicit frequencies -> |Ex|^2
               map just behind the exit face of the resonator
               (reproduces Fig. 2(c) / Fig. 3(c))

Read README.md before running: default resolution is a *fast smoke-test*
setting, not a production setting. A faithful run (features down to 20 nm
resolved with ~8-10 points) needs resolution on the order of a few
hundred (1/um) and should be run in parallel (mpirun) on a cluster; wall
time and memory both scale steeply with resolution in 3D.

Example (reproducing Fig. 2(b), the 1:2 position):
    mpirun -np 4 python3 simulate.py --mode spectrum --empty \
        --outdir results/pos12
    mpirun -np 4 python3 simulate.py --mode spectrum --rod none \
        --outdir results/pos12
    mpirun -np 4 python3 simulate.py --mode spectrum --rod large \
        --position 1:2 --outdir results/pos12
    mpirun -np 4 python3 simulate.py --mode spectrum --rod small \
        --position 1:2 --outdir results/pos12
    python3 analyze.py spectrum --indir results/pos12 --outfile fig2b.png

See run_all.sh for the full set of runs needed for Figs. 2 and 3, and
README.md for the modeling assumptions and their caveats.
"""

import argparse
import os

import numpy as np

import meep as mp

from geometry import ResonatorParams, RodParams, build_geometry, rod_y_center

ROD_WIDTHS = {"large": 0.060, "small": 0.020}  # um


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["spectrum", "field"], required=True)
    p.add_argument("--rod", choices=["none", "large", "small"], default="none")
    p.add_argument("--position", choices=["1:2", "1:1"], default="1:2",
                    help="d1:d2 position of the rod along the slot (ignored if --rod none)")
    p.add_argument("--film-metal", default="Au", choices=["Au", "Pt"])
    p.add_argument("--rod-metal", default="Pt", choices=["Au", "Pt"])
    p.add_argument("--custom-drude-film", type=str, default=None,
                    help="'f_p,gamma' in 1/um to override the film material with a plain "
                         "single-pole Drude medium (e.g. from Ordal et al. values) instead "
                         "of meep.materials.Au/Pt")
    p.add_argument("--custom-drude-rod", type=str, default=None,
                    help="'f_p,gamma' in 1/um to override the rod material similarly")

    p.add_argument("--l1", type=float, default=10.0, help="slot length, um")
    p.add_argument("--w1", type=float, default=0.020, help="slot width, um")
    p.add_argument("--t1", type=float, default=0.020, help="film thickness, um")
    p.add_argument("--l2", type=float, default=0.30, help="rod length (across slot), um")
    p.add_argument("--t2", type=float, default=0.020, help="rod thickness, um")

    p.add_argument("--resolution", type=float, default=60,
                    help="pixels per um. >=250 recommended for a faithful run; "
                         "default is a fast/low-fidelity smoke-test value.")
    p.add_argument("--dpml", type=float, default=1.0, help="PML thickness, um")
    p.add_argument("--pad-x", type=float, default=0.30, help="vacuum padding beyond the rod, x, um")
    p.add_argument("--pad-y", type=float, default=1.0, help="vacuum padding beyond the slot, y, um")
    p.add_argument("--pad-z", type=float, default=2.0, help="vacuum padding above/below film, z, um")

    p.add_argument("--fcen", type=float, default=0.09, help="source center frequency (1/um)")
    p.add_argument("--df", type=float, default=0.16, help="source frequency width (1/um)")
    p.add_argument("--nfreq", type=int, default=400, help="number of flux frequency points (spectrum mode)")
    p.add_argument("--freqs", type=str, default=None,
                    help="comma-separated frequencies in 1/um for field mode, e.g. '0.05,0.15'")
    p.add_argument("--field-z-offset", type=float, default=0.02,
                    help="distance below the film's bottom face where the field-map plane sits, um")

    p.add_argument("--empty", action="store_true",
                    help="run with no film/slot/rod at all (vacuum propagation), to get the "
                         "incident-flux reference P_inc(f) that spectrum-mode transmittance "
                         "is normalized against. Use with --mode spectrum.")
    p.add_argument("--no-symmetry", action="store_true", help="disable mirror-symmetry speedups")
    p.add_argument("--decay-tol", type=float, default=1e-4)
    p.add_argument("--outdir", type=str, default="results")
    p.add_argument("--tag", type=str, default=None, help="override the auto-generated run tag")
    return p.parse_args()


def make_run_tag(args):
    if args.tag:
        return args.tag
    if args.empty:
        return "reference"
    if args.rod == "none":
        return "norod"
    pos = args.position.replace(":", "")
    return f"{args.rod}_pos{pos}"


def _parse_drude_pair(s):
    if s is None:
        return None
    f_p, gamma = (float(x) for x in s.split(","))
    return (f_p, gamma)


def build_sim(args):
    res = ResonatorParams(
        l1=args.l1, w1=args.w1, t1=args.t1, film_metal=args.film_metal,
        film_custom_drude=_parse_drude_pair(args.custom_drude_film),
    )

    rod = None
    if args.rod != "none":
        ratio = (1.0, 2.0) if args.position == "1:2" else (1.0, 1.0)
        rod = RodParams(
            s=ROD_WIDTHS[args.rod],
            l2=args.l2,
            t2=args.t2,
            rod_metal=args.rod_metal,
            rod_custom_drude=_parse_drude_pair(args.custom_drude_rod),
            position_ratio=ratio,
        )

    sx = args.w1 + args.l2 + 2 * args.pad_x + 2 * args.dpml
    sy = args.l1 + 2 * args.pad_y + 2 * args.dpml
    sz = args.t1 + 2 * args.pad_z + 2 * args.dpml
    cell = mp.Vector3(sx, sy, sz)

    inner_x = sx - 2 * args.dpml
    inner_y = sy - 2 * args.dpml

    z_src = sz / 2.0 - args.dpml - 0.3 * args.pad_z  # source plane above the film, inside the padding
    z_mon = -sz / 2.0 + args.dpml + 0.3 * args.pad_z  # transmission-flux plane below the film

    geometry = [] if args.empty else build_geometry(res, rod, cell_x=sx, cell_y=sy)

    sources = [
        mp.Source(
            mp.GaussianSource(frequency=args.fcen, fwidth=args.df, is_integrated=True),
            component=mp.Ex,
            center=mp.Vector3(0, 0, z_src),
            size=mp.Vector3(inner_x, inner_y, 0),
        )
    ]

    symmetries = []
    if not args.no_symmetry:
        symmetries.append(mp.Mirror(mp.X, phase=-1))
        if rod is None or args.position == "1:1":
            symmetries.append(mp.Mirror(mp.Y, phase=+1))

    sim = mp.Simulation(
        cell_size=cell,
        resolution=args.resolution,
        boundary_layers=[mp.PML(args.dpml)],
        geometry=geometry,
        sources=sources,
        symmetries=symmetries,
        default_material=mp.Medium(epsilon=1.0),
    )

    return sim, dict(
        sx=sx, sy=sy, sz=sz, inner_x=inner_x, inner_y=inner_y,
        z_src=z_src, z_mon=z_mon, res=res, rod=rod,
    )


def run_spectrum(args):
    sim, geo = build_sim(args)

    trans_region = mp.FluxRegion(
        center=mp.Vector3(0, 0, geo["z_mon"]),
        size=mp.Vector3(geo["inner_x"], geo["inner_y"], 0),
    )
    trans_flux = sim.add_flux(args.fcen, args.df, args.nfreq, trans_region)

    monitor_pt = mp.Vector3(0, 0, geo["z_mon"])
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, monitor_pt, args.decay_tol))

    freqs = np.array(mp.get_flux_freqs(trans_flux))
    flux = np.array(mp.get_fluxes(trans_flux))

    os.makedirs(args.outdir, exist_ok=True)
    tag = make_run_tag(args)
    outfile = os.path.join(args.outdir, f"flux_{tag}.npz")
    if mp.am_master():
        np.savez(outfile, freqs=freqs, flux=flux, args=vars(args))
        print(f"[simulate.py] wrote {outfile}")


def run_field(args):
    if not args.freqs:
        raise SystemExit("--freqs is required in field mode, e.g. --freqs 0.05,0.15")
    freqs = [float(x) for x in args.freqs.split(",")]

    sim, geo = build_sim(args)

    field_plane_z = -geo["res"].t1 / 2.0 - args.field_z_offset
    plane = mp.Volume(
        center=mp.Vector3(0, 0, field_plane_z),
        size=mp.Vector3(geo["inner_x"], geo["inner_y"], 0),
    )
    # One add_dft_fields call per target frequency (nfreq=1, df~0), rather
    # than relying on the less-common "explicit freq list" overload -- this
    # sticks to the long-standing (fcen, df, nfreq) calling convention that
    # add_flux/add_dft_fields have always supported.
    dft_objs = [
        sim.add_dft_fields([mp.Ex], f, 0, 1, where=plane)
        for f in freqs
    ]

    monitor_pt = mp.Vector3(0, 0, plane.center.z)
    sim.run(until_after_sources=mp.stop_when_fields_decayed(
        50, mp.Ex, monitor_pt, args.decay_tol))

    os.makedirs(args.outdir, exist_ok=True)
    tag = make_run_tag(args)
    for f, dft_obj in zip(freqs, dft_objs):
        ex = sim.get_dft_array(dft_obj, mp.Ex, 0)
        intensity = np.abs(ex) ** 2
        outfile = os.path.join(args.outdir, f"field_{tag}_f{f:.4f}.npy")
        if mp.am_master():
            np.save(outfile, intensity)
            print(f"[simulate.py] wrote {outfile}  shape={intensity.shape}")


def main():
    args = parse_args()
    if args.mode == "spectrum":
        run_spectrum(args)
    else:
        run_field(args)


if __name__ == "__main__":
    main()
