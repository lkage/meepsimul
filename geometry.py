"""Geometry builder for the THz pinch-harmonics nanoresonator + nano rod.

Coordinate convention (matches Fig. 1(a) of the paper):
    z : propagation direction of the normally-incident THz pulse
    x : E-field polarization direction == the SHORT (width) axis of the
        slot aperture, w1. The incident field is "perpendicular to the
        long axis of the rectangle" -> E is along x.
    y : the LONG axis of the slot aperture, l1. The nano rod's position
        along the resonator ("1:2" or "1:1") is a position along y.

The gold film is modeled as an (effectively) infinite sheet of thickness
t1 in z, pierced by a single finite rectangular slot of length l1 (along
y) and width w1 (along x) -- i.e. a single nanoresonator, not a periodic
grating, matching "assuming a rectangular hole ... in a thin gold film".

The nano rod sits on top of the film, oriented with its long dimension
l2 along x (bridging across the w1-wide slot, as in the SEM images of
Fig. 1(b)) and its "width" s along y (this is the dimension that must be
below the metal skin depth to produce the pinch-harmonic effect, and is
the free parameter in Fig. 4 of the paper). Its center is placed at
fractional position (d1 : d2) along the slot length l1, measured from the
y = -l1/2 edge.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import meep as mp

from materials import get_metal


@dataclass
class ResonatorParams:
    l1: float = 10.0        # slot length (um), along y
    w1: float = 0.020       # slot width (um), along x
    t1: float = 0.020       # gold film thickness (um), along z
    film_metal: str = "Au"
    film_custom_drude: Optional[Tuple[float, float]] = None  # (f_p, gamma) override


@dataclass
class RodParams:
    s: float = 0.060        # rod width (um), along y -- the swept parameter
    l2: float = 0.30        # rod length (um), along x (must exceed w1)
    t2: float = 0.020       # rod thickness (um), along z
    rod_metal: str = "Pt"
    rod_custom_drude: Optional[Tuple[float, float]] = None  # (f_p, gamma) override
    position_ratio: Tuple[float, float] = (1.0, 2.0)  # (d1 : d2) along l1


def rod_y_center(res: ResonatorParams, rod: RodParams) -> float:
    """y-coordinate of the rod center for a given d1:d2 split of l1,
    measured from the slot center (y=0 at the middle of the resonator).
    d1 is measured from the y = -l1/2 edge.
    """
    d1_frac = rod.position_ratio[0] / sum(rod.position_ratio)
    d1 = d1_frac * res.l1
    return d1 - res.l1 / 2.0


def build_geometry(
    res: ResonatorParams,
    rod: Optional[RodParams],
    cell_x: float,
    cell_y: float,
):
    """Return a Meep geometry list for the given transverse cell size.

    The gold film and (optional) rod are built as simple mp.Block objects;
    the slot is a vacuum block placed after (i.e. carved out of) the gold
    block, exploiting Meep's "later objects take precedence" geometry
    rule. cell_x/cell_y are the full simulation-cell transverse extents;
    the film is sized to match them exactly so it extends all the way
    into the PML on every transverse side (standard Meep practice for
    representing a large/infinite sheet, avoiding an artificial vacuum
    gap -- and the edge diffraction that would come with it -- between
    the film and the absorbing boundary)."""

    gold = get_metal(res.film_metal, custom_drude=res.film_custom_drude)
    vacuum = mp.Medium(epsilon=1.0)

    film = mp.Block(
        size=mp.Vector3(cell_x, cell_y, res.t1),
        center=mp.Vector3(0, 0, 0),
        material=gold,
    )

    slot = mp.Block(
        size=mp.Vector3(res.w1, res.l1, res.t1 + 1e-3),  # tiny z-oversize avoids coincident faces
        center=mp.Vector3(0, 0, 0),
        material=vacuum,
    )

    geometry = [film, slot]

    if rod is not None:
        rod_metal = get_metal(rod.rod_metal, custom_drude=rod.rod_custom_drude)
        y0 = rod_y_center(res, rod)
        z0 = res.t1 / 2.0 + rod.t2 / 2.0  # sits directly on top of the film
        rod_block = mp.Block(
            size=mp.Vector3(rod.l2, rod.s, rod.t2),
            center=mp.Vector3(0, y0, z0),
            material=rod_metal,
        )
        geometry.append(rod_block)

    return geometry
