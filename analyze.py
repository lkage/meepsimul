#!/usr/bin/env python3
"""Post-processing / plotting for simulate.py output.

Two subcommands:

  spectrum  Load flux_*.npz files from a directory (produced by
            `simulate.py --mode spectrum`), normalize by the reference
            (empty-cell) run, and overlay the "without rod / large rod /
            small rod" transmittance curves -- reproducing the style of
            Fig. 2(b) / Fig. 3(b) of the paper. Also reports the
            strongest peak frequencies it finds (candidate 1st/3rd
            harmonic modes).

  field     Load field_*.npy files (produced by
            `simulate.py --mode field`, |Ex|^2 maps) and plot them as
            false-color images, reproducing the style of Fig. 2(c) /
            Fig. 3(c).

Usage:
    python3 analyze.py spectrum --indir results/pos12 --outfile fig2b.png
    python3 analyze.py field --indir results/pos12 --pattern 'field_*_f0.05*.npy' \
        --outfile fig2c_fundamental.png
"""

import argparse
import glob
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def cmd_spectrum(args):
    ref_file = os.path.join(args.indir, "flux_reference.npz")
    if not os.path.exists(ref_file):
        raise SystemExit(
            f"missing {ref_file}. Run `simulate.py --mode spectrum --empty` first "
            f"to produce the incident-flux reference."
        )
    ref = np.load(ref_file, allow_pickle=True)
    ref_freqs, ref_flux = ref["freqs"], ref["flux"]

    labels = {
        "norod": "without nano rod",
        "large_pos12": "600 nm-width nano rod",
        "small_pos12": "250 nm-width nano rod",
        "large_pos11": "600 nm-width nano rod",
        "small_pos11": "250 nm-width nano rod",
    }

    fig, ax = plt.subplots(figsize=(6, 4.5))
    found_any = False
    for tag, label in labels.items():
        f = os.path.join(args.indir, f"flux_{tag}.npz")
        if not os.path.exists(f):
            continue
        found_any = True
        d = np.load(f, allow_pickle=True)
        freqs, flux = d["freqs"], d["flux"]
        if not np.allclose(freqs, ref_freqs):
            raise SystemExit(f"{f} has different frequency grid than the reference run")
        T = flux / ref_flux
        ax.plot(freqs, T, label=label)

        # crude peak report
        try:
            from scipy.signal import find_peaks

            idx, _ = find_peaks(T, height=T.max() * 0.15)
            if len(idx):
                peak_f = freqs[idx][np.argsort(T[idx])[::-1]]
                print(f"[{tag}] candidate resonance frequencies (1/um), "
                      f"strongest first: {np.round(peak_f[:5], 4).tolist()}")
        except ImportError:
            pass

    if not found_any:
        raise SystemExit(f"no flux_*.npz runs found in {args.indir} (besides the reference)")

    ax.set_xlabel("Frequency (1/um, Meep units)")
    ax.set_ylabel("Transmittance")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outfile, dpi=150)
    print(f"[analyze.py] wrote {args.outfile}")


def cmd_field(args):
    files = sorted(glob.glob(os.path.join(args.indir, args.pattern)))
    if not files:
        raise SystemExit(f"no files matched {args.pattern!r} in {args.indir}")

    n = len(files)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
    vmax = max(np.load(f).max() for f in files)
    for ax, f in zip(axes[0], files):
        data = np.load(f)  # shape (nx, ny): |Ex|^2 on the monitor plane
        im = ax.imshow(data.T, origin="lower", aspect="auto", cmap="turbo", vmin=0, vmax=vmax)
        ax.set_title(os.path.basename(f), fontsize=8)
        ax.set_xlabel("x index (width)")
        ax.set_ylabel("y index (length)")
    fig.colorbar(im, ax=axes[0].tolist(), label="|Ex|^2 (a.u.)", shrink=0.8)
    fig.savefig(args.outfile, dpi=150)
    print(f"[analyze.py] wrote {args.outfile}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("spectrum")
    ps.add_argument("--indir", required=True)
    ps.add_argument("--outfile", default="spectrum.png")
    ps.set_defaults(func=cmd_spectrum)

    pf = sub.add_parser("field")
    pf.add_argument("--indir", required=True)
    pf.add_argument("--pattern", default="field_*.npy")
    pf.add_argument("--outfile", default="field.png")
    pf.set_defaults(func=cmd_field)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
