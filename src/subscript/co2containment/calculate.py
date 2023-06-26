from dataclasses import dataclass
from typing import List, Union, Literal, Optional

import numpy as np
from shapely.geometry import Polygon, MultiPolygon
from subscript.co2containment.co2_calculation import Co2Data
from subscript.co2containment.co2_calculation import CalculationType


@dataclass
class ContainedCo2:
    """
    Dataclass with amount of Co2 in/out a given area for a given phase
    at different time steps

    Args:
        date (str): A given time step
        amount (float): Numerical value with the computed amount at "date"
        phase (Literal): One of gas/aqueous/undefined. The phase of "amount".
        location (Literal): One of contained/outside/hazardous. The location "amount"
                            corresponds to
        zone (str): 
        
    """
    date: str
    amount: float
    phase: Literal["gas", "aqueous", "undefined"]
    location: Literal["contained", "outside", "hazardous"]
    zone: Optional[str] = None

    def __post_init__(self):
        """
        If the slot "data" of a ContainedCo2 object does not contain "-", this
        function converts it to the format yyyy-mm-dd
        
        """
        if "-" not in self.date:
            d = self.date
            self.date = f"{d[:4]}-{d[4:6]}-{d[6:]}"


def calculate_co2_containment(
        co2_data: Co2Data,
        containment_polygon: Union[Polygon, MultiPolygon],
        hazardous_polygon: Union[Polygon, MultiPolygon, None],
        calc_type: CalculationType
) -> List[ContainedCo2]:
    """
    Calculates the amount (mass/volume) of CO2 within given boundaries (contained/outside/hazardous)
    at each time step for each phase (aqueous/gaseous). Result is a list of ContainedCo2 objects.

    Args:
        co2_data (Co2Data): Information of the amount of CO2 at each cell in each time step
        containment_polygon (Union[Polygon,Multipolygon]): The polygon that defines the containment
                                                           area
        hazardous_polygon (Union[Polygon,Multipolygon]): The polygon that defines the hazardous
                                                         area   
        calc_type (CalculationType): Which calculation is to be performed (mass / volume_extent / 
                                     volume_actual / volume_actual_simple)

    Returns:
        List[ContainedCo2]
    """
    if containment_polygon is not None:
        is_contained = _calculate_containment(co2_data.x, co2_data.y, containment_polygon)
    else:
        is_contained = np.array([True]*len(co2_data.x))
    if hazardous_polygon is not None:
        is_hazardous = _calculate_containment(co2_data.x, co2_data.y, hazardous_polygon)
    else:
        is_hazardous = np.array([False]*len(co2_data.x))
    # Count as hazardous if the two boundaries overlap:
    is_contained = [x if not y else False for x, y in zip(is_contained, is_hazardous)]
    is_outside = [not x and not y for x, y in zip(is_contained, is_hazardous)]
    if co2_data.zone is None:
        if calc_type == CalculationType.volume_extent:
            return [
                c
                for w in co2_data.data_list
                for c in [
                    ContainedCo2(w.date, sum(w.volume_coverage[is_contained]), "undefined", "contained"),
                    ContainedCo2(w.date, sum(w.volume_coverage[is_outside]), "undefined", "outside"),
                    ContainedCo2(w.date, sum(w.volume_coverage[is_hazardous]), "undefined", "hazardous"),
                ]]
        else:
            return [
                c
                for w in co2_data.data_list
                for c in [
                    ContainedCo2(w.date, sum(w.gas_phase[is_contained]), "gas", "contained"),
                    ContainedCo2(w.date, sum(w.gas_phase[is_outside]), "gas", "outside"),
                    ContainedCo2(w.date, sum(w.gas_phase[is_hazardous]), "gas", "hazardous"),
                    ContainedCo2(w.date, sum(w.aqu_phase[is_contained]), "aqueous", "contained"),
                    ContainedCo2(w.date, sum(w.aqu_phase[is_outside]), "aqueous", "outside"),
                    ContainedCo2(w.date, sum(w.aqu_phase[is_hazardous]), "aqueous", "hazardous"),
                ]
            ]
    else:
        zone_map = {z: co2_data.zone == z for z in np.unique(co2_data.zone)}
        if calc_type == CalculationType.volume_extent:
            return [
                c
                for w in co2_data.data_list
                for zn, zm in zone_map.items()
                for c in [
                    ContainedCo2(
                        w.date, sum(w.volume_coverage[is_contained & zm]), "gas", "contained", zn
                    ),
                    ContainedCo2(
                        w.date, sum(w.volume_coverage[is_outside & zm]), "gas", "outside", zn
                    ),
                    ContainedCo2(
                        w.date, sum(w.volume_coverage[is_hazardous & zm]), "gas", "hazardous", zn
                    ),
                ]
            ]
        else:
            return [
                c
                for w in co2_data.data_list
                for zn, zm in zone_map.items()
                for c in [
                    ContainedCo2(
                        w.date, sum(w.gas_phase[is_contained & zm]), "gas", "contained", zn
                    ),
                    ContainedCo2(
                        w.date, sum(w.gas_phase[is_outside & zm]), "gas", "outside", zn
                    ),
                    ContainedCo2(
                        w.date, sum(w.gas_phase[is_hazardous & zm]), "gas", "hazardous", zn
                    ),
                    ContainedCo2(
                        w.date, sum(w.aqu_phase[is_contained & zm]), "aqueous", "contained", zn
                    ),
                    ContainedCo2(
                        w.date, sum(w.aqu_phase[is_outside & zm]), "aqueous", "outside", zn
                    ),
                    ContainedCo2(
                        w.date, sum(w.aqu_phase[is_hazardous & zm]), "aqueous", "hazardous", zn
                    ),
                ]
            ]


def _calculate_containment(
    x: np.ndarray,
    y: np.ndarray,
    poly: Union[Polygon, MultiPolygon]
) -> np.ndarray:
    """
    Determines if (x,y) coordinates belong to a given polygon.

    Args:
        x (np.ndarray): x coordinates
        y (np.ndarray): y coordinates
        poly (Union[Polygon, MultiPolygon]): The polygon that determines the 
                                             containment of the (x,y) coordinates

    Returns:
        np.ndarray    
    """
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
