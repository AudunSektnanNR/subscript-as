from dataclasses import dataclass
from typing import List, Union, Literal, Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from subscript.co2containment.co2_mass_calculation.co2_mass_calculation import Co2MassData


@dataclass
class ContainedCo2:
    date: str
    amount_kg: float
    phase: Literal["gas", "aqueous"]
    inside_containment_boundary: bool
    outside_boundaries: bool
    inside_hazardous_boundary: bool
    zone: Optional[str] = None

    def __post_init__(self):
        if "-" not in self.date:
            d = self.date
            self.date = f"{d[:4]}-{d[4:6]}-{d[6:]}"


def calculate_co2_containment(
    co2_mass_data: Co2MassData,
    containment_polygon: Union[Polygon, MultiPolygon],
    hazardous_polygon: Union[Polygon, MultiPolygon]
) -> List[ContainedCo2]:
    is_contained = _calculate_containment(co2_mass_data.x, co2_mass_data.y, containment_polygon)
    is_hazardous = _calculate_containment(co2_mass_data.x, co2_mass_data.y, hazardous_polygon)
    # Count as hazardous if the two boundaries overlap:
    is_contained = [x if not y else False for x, y in zip(is_contained, is_hazardous)]
    is_outside = [not x and not y for x, y in zip(is_contained, is_hazardous)] 
    if co2_mass_data.zone is None:
        return [
            c
            for w in co2_mass_data.data_list
            # for c in [
            #     ContainedCo2(w.date, w.gas_phase_kg[~is_contained].sum(), "gas", False),
            #     ContainedCo2(w.date, w.gas_phase_kg[is_contained].sum(), "gas", True),
            #     ContainedCo2(w.date, w.aqu_phase_kg[~is_contained].sum(), "aqueous", False),
            #     ContainedCo2(w.date, w.aqu_phase_kg[is_contained].sum(), "aqueous", True),
            # ]
            for c in [
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_contained]), "gas", True, False, False),
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_outside]), "gas", False, True, False),
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_hazardous]), "gas", False, False, True),
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_contained]), "aqueous", True, False, False),
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_outside]), "aqueous", False, True, False),
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_hazardous]), "aqueous", False, False, True),
            ]
        ]
    else:
        zone_map = {z: co2_mass_data.zone == z for z in np.unique(co2_mass_data.zone)}
        return [
            c
            for w in co2_mass_data.data_list
            for zn, zm in zone_map.items()
            for c in [
                ContainedCo2(
                    w.date, sum(w.gas_phase_kg[~is_contained & zm]), "gas", False, zn
                ),
                ContainedCo2(
                    w.date, sum(w.gas_phase_kg[(is_contained) & zm]), "gas", True, zn
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase_kg[~is_contained & zm]), "aqueous", False, zn
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase_kg[(is_contained) & zm]), "aqueous", True, zn
                ),
            ]
        ]


def _calculate_containment(
    x: np.ndarray,
    y: np.ndarray,
    poly: Union[Polygon, MultiPolygon]
) -> np.ndarray:
    try:
        import pygeos
        points = pygeos.points(x, y)
        poly = pygeos.from_shapely(poly)
        return pygeos.contains(poly, points)
    except ImportError:
        import shapely.geometry as sg
        return np.array([
            poly.contains(sg.Point(_x, _y))
            for _x, _y in zip(x, y)
        ])
