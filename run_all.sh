#!/usr/bin/env bash
# Runs the full set of simulations needed to reproduce Fig. 2(b) (rod at
# the 1:2 position) and Fig. 3(b) (rod at the 1:1 midpoint) of the paper,
# then plots the overlaid transmittance spectra.
#
# WARNING: the defaults here (RESOLUTION=60) are a fast smoke test, not a
# faithful reproduction -- see README.md. For a real run, raise
# RESOLUTION (250+) and run under mpirun with many more ranks; expect
# each run to take a long time and a lot of memory in 3D.
set -euo pipefail

NP="${NP:-4}"                 # MPI ranks
RESOLUTION="${RESOLUTION:-60}"
NFREQ="${NFREQ:-400}"
COMMON="--resolution $RESOLUTION --nfreq $NFREQ"

run() {
  echo "+++ $*"
  mpirun -np "$NP" python3 simulate.py "$@"
}

for POS_DIR in pos12 pos11; do
  if [ "$POS_DIR" = pos12 ]; then POSITION=1:2; else POSITION=1:1; fi
  OUTDIR="results/$POS_DIR"
  mkdir -p "$OUTDIR"

  # 1) incident-flux reference (no structure at all)
  run --mode spectrum --empty $COMMON --outdir "$OUTDIR"

  # 2) resonator alone, no rod
  run --mode spectrum --rod none $COMMON --outdir "$OUTDIR"

  # 3) 600 nm-wide ("large") Pt rod at this position
  run --mode spectrum --rod large --position "$POSITION" $COMMON --outdir "$OUTDIR"

  # 4) 250 nm-wide ("small") Pt rod at this position
  run --mode spectrum --rod small --position "$POSITION" $COMMON --outdir "$OUTDIR"

  # overlay plot + printed candidate resonance frequencies
  python3 analyze.py spectrum --indir "$OUTDIR" --outfile "$OUTDIR/spectrum.png"
done

echo
echo "Spectra written to results/pos12/spectrum.png and results/pos11/spectrum.png."
echo "To reproduce the |Ex|^2 field maps (Fig. 2(c)/3(c)), read off the printed"
echo "1st/3rd-harmonic candidate frequencies above and run, e.g.:"
echo
echo "  mpirun -np \$NP python3 simulate.py --mode field --rod none \\"
echo "      --freqs <f1>,<3f1> --outdir results/pos12"
echo "  mpirun -np \$NP python3 simulate.py --mode field --rod large --position 1:2 \\"
echo "      --freqs <f1>,<3f1> --outdir results/pos12"
echo "  mpirun -np \$NP python3 simulate.py --mode field --rod small --position 1:2 \\"
echo "      --freqs <f1>,<3f1> --outdir results/pos12"
echo "  python3 analyze.py field --indir results/pos12 --pattern 'field_*.npy' \\"
echo "      --outfile results/pos12/fields.png"
