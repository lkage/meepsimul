"""Metal media for the THz pinch-harmonics reproduction.

Rather than hand-transcribing Drude/Lorentz-Drude coefficients into this
file (a step that risks silent numeric errors), the default path here
imports Meep's own bundled, literature-referenced material library
(meep.materials.Au / meep.materials.Pt, a 6-pole Lorentz-Drude fit to
Rakic et al., Appl. Opt. 37, 5271 (1998)) and uses it as-is. Whatever
values ship with your installed Meep are the values that get used --
nothing here duplicates or could drift from them.

IMPORTANT CAVEAT: Meep's Au/Pt fits are calibrated against data from
roughly 0.2-12 um wavelength (near-IR to UV), not against far-IR/THz
data. The paper itself models gold and platinum with a plain Drude term
in this regime (citing Ordal et al., Appl. Opt. 22, 1099 (1983), Ref.
[22] of the paper). In the THz band the interband (Lorentzian) poles of
the Rakic fit contribute negligibly and the response is dominated by the
free-electron Drude pole, so using meep.materials.Au/Pt here is a
physically reasonable extrapolation and keeps the model "audited" (it is
exactly what ships with Meep, nothing invented) -- but it is still an
extrapolation, not a THz-band-validated fit. For a quantitatively exact
reproduction, get Ordal's actual tabulated omega_p/omega_tau numbers
from the paper's Ref. [22] and pass them via --custom-drude-film /
--custom-drude-rod (see simulate.py --help), which builds a plain
single-pole Drude medium from your own numbers instead.
"""

import meep as mp
import meep.materials as mm

_BUILTIN = {"Au": mm.Au, "Pt": mm.Pt}


def get_metal(name, custom_drude=None):
    """Return an mp.Medium for the named metal.

    custom_drude: optional (f_p, gamma) tuple, both in Meep frequency
    units (1/um, with the length unit a = 1 um used throughout this
    project). If given, builds a plain single-pole Drude medium
    eps(w) = 1 - f_p^2/(w^2 + i*w*gamma) instead of using the built-in
    library material.
    """
    if custom_drude is not None:
        f_p, gamma = custom_drude
        return mp.Medium(
            epsilon=1.0,
            E_susceptibilities=[mp.DrudeSusceptibility(frequency=f_p, gamma=gamma, sigma=1)],
        )
    if name not in _BUILTIN:
        raise ValueError(f"no built-in metal '{name}'; supported: {list(_BUILTIN)} "
                          f"(or pass --custom-drude-* with your own f_p,gamma)")
    return _BUILTIN[name]
