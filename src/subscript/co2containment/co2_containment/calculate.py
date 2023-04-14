from dataclasses import dataclass
from typing import List, Union, Literal, Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon

from subscript.co2containment.co2_mass_calculation.co2_mass_calculation import Co2MassData
from subscript.co2containment.co2_mass_calculation.co2_mass_calculation import Co2VolumeData


@dataclass
class ContainedCo2:
    date: str
    amount_kg: float
    phase: Literal["gas", "aqueous"]
    location: Literal["contained", "outside", "hazardous"]
    zone: Optional[str] = None

    def __post_init__(self):
        if "-" not in self.date:
            d = self.date
            self.date = f"{d[:4]}-{d[4:6]}-{d[6:]}"

@dataclass
class ContainedCo2Vol:
    date: str
    amount_m3: float
    phase: Literal["gas", "aqueous","extent"]
    location: Literal["contained", "outside", "hazardous"]
    zone: Optional[str] = None
    def __post_init__(self):
        if "-" not in self.date:
            d = self.date
            self.date = f"{d[:4]}-{d[4:6]}-{d[6:]}"

def calculate_co2_containment(
    co2_mass_data: Co2MassData,
    containment_polygon: Union[Polygon, MultiPolygon],
    hazardous_polygon: Union[Polygon, MultiPolygon, None]
) -> List[ContainedCo2]:
    is_contained = _calculate_containment(co2_mass_data.x, co2_mass_data.y, containment_polygon)
    if hazardous_polygon is not None:
        is_hazardous = _calculate_containment(co2_mass_data.x, co2_mass_data.y, hazardous_polygon)
    else:
        is_hazardous = np.array([False]*len(is_contained))
    # Count as hazardous if the two boundaries overlap:
    is_contained = [x if not y else False for x, y in zip(is_contained, is_hazardous)]
    is_outside = [not x and not y for x, y in zip(is_contained, is_hazardous)] 
    if co2_mass_data.zone is None:
        return [
            c
            for w in co2_mass_data.data_list
            for c in [
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_contained]), "gas", "contained"),
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_outside]), "gas", "outside"),
                ContainedCo2(w.date, sum(w.gas_phase_kg[is_hazardous]), "gas", "hazardous"),
                ContainedCo2(w.date, sum(w.aqu_phase_kg[is_contained]), "aqueous", "contained"),
                ContainedCo2(w.date, sum(w.aqu_phase_kg[is_outside]), "aqueous", "outside"),
                ContainedCo2(w.date, sum(w.aqu_phase_kg[is_hazardous]), "aqueous", "hazardous"),
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
                    w.date, sum(w.gas_phase_kg[is_contained & zm]), "gas", "contained", zn
                ),
                ContainedCo2(
                    w.date, sum(w.gas_phase_kg[(is_outside) & zm]), "gas", "outside", zn
                ),
                ContainedCo2(
                    w.date, sum(w.gas_phase_kg[(is_hazardous) & zm]), "gas", "hazardous", zn
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase_kg[is_contained & zm]), "gaaqueouss", "contained", zn
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase_kg[(is_outside) & zm]), "aqueous", "outside", zn
                ),
                ContainedCo2(
                    w.date, sum(w.aqu_phase_kg[(is_hazardous) & zm]), "aqueous", "hazardous", zn
                ),
            ]
        ]

def calculate_co2_containment_vol(
    co2_vol_data: Co2VolumeData,
    containment_polygon: Union[Polygon, MultiPolygon],
    hazardous_polygon: Union[Polygon, MultiPolygon, None]
) -> List[ContainedCo2Vol]:
    is_contained = {p.property:{t.date:_calculate_containment(t.x, t.y, containment_polygon) for t in p.data_list} for p in co2_vol_data.data_list}
    if hazardous_polygon is not None:
	    is_hazardous = {p.property:{t.date:_calculate_containment(t.x, t.y, hazardous_polygon) for t in p.data_list} for p in co2_vol_data.data_list}
    else:
        is_hazardous = {p:{t: [False]*len(is_contained[p][t]) for t in is_contained[p]} for p in is_contained}
    # Count as hazardous if the two boundaries overlap:
    is_contained = {p:{t:[x if not y else False for x,y in zip(is_contained[p][t],is_hazardous[p][t])]
                       for t in is_contained[p]}
                    for p in is_contained}
    is_outside = {p:{t:[not x and not y for x, y in zip(is_contained[p][t],is_hazardous[p][t])]
                     for t in is_contained[p]}
                  for p in is_contained}
    if co2_vol_data.zone is None:
        return {p.property: [
            c
            for w in p.data_list
            for c in [
                ContainedCo2Vol(w.date, sum(w.gas_phase_m3[is_contained[p.property][w.date]]), "gas", "contained"),
                ContainedCo2Vol(w.date, sum(w.gas_phase_m3[is_outside[p.property][w.date]]), "gas", "outside"),
                ContainedCo2Vol(w.date, sum(w.gas_phase_m3[is_hazardous[p.property][w.date]]), "gas", "hazardous"),
                ContainedCo2Vol(w.date, sum(w.aqu_phase_m3[is_contained[p.property][w.date]]), "aqueous", "contained"),
                ContainedCo2Vol(w.date, sum(w.aqu_phase_m3[is_outside[p.property][w.date]]), "aqueous", "outside"),
                ContainedCo2Vol(w.date, sum(w.aqu_phase_m3[is_hazardous[p.property][w.date]]), "aqueous", "hazardous"),
                ContainedCo2Vol(w.date, sum(w.volume_coverage[is_contained[p.property][w.date]]), "extent", "contained"),
                ContainedCo2Vol(w.date, sum(w.volume_coverage[is_outside[p.property][w.date]]), "extent", "outside"),
                ContainedCo2Vol(w.date, sum(w.volume_coverage[is_hazardous[p.property][w.date]]), "extent", "hazardous"),
            ]] for p in co2_vol_data.data_list}
    else:
        zone_map = {z: co2_vol_data.zone == z for z in np.unique(co2_vol_data.zone)}
        return {p.property: [
            c
            for w in p.data_list
            for zn, zm in zone_map.items()
            for c in [
                ContainedCo2Vol(
                    w.date, sum(w.gas_phase_m3[is_contained[p.property][w.date] & zm]), "gas", "contained", zn
                ),
                ContainedCo2Vol(
                    w.date, sum(w.gas_phase_m3[(is_outside[p.property][w.date]) & zm]), "gas", "outside", zn
                ),
                ContainedCo2Vol(
                    w.date, sum(w.gas_phase_m3[(is_hazardous[p.property][w.date]) & zm]), "gas", "hazardous", zn
                ),
                ContainedCo2Vol(
                    w.date, sum(w.aqu_phase_m3[is_contained[p.property][w.date] & zm]), "gaaqueouss", "contained", zn
                ),
                ContainedCo2Vol(
                    w.date, sum(w.aqu_phase_m3[(is_outside[p.property][w.date]) & zm]), "aqueous", "outside", zn
                ),
                ContainedCo2Vol(
                    w.date, sum(w.aqu_phase_m3[(is_hazardous[p.property][w.date]) & zm]), "aqueous", "hazardous", zn
                ),
            ]
        ] for p in co2_vol_data.data_list}
  
 

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
