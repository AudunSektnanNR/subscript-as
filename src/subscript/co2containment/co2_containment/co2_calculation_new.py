from dataclasses import dataclass, fields
from enum import Enum
from typing import Dict, List, Optional,Literal, Tuple
import sys

import numpy as np
import xtgeo
from ecl.eclfile import EclFile
from ecl.grid import EclGrid

DEFAULT_CO2_MOLAR_MASS = 44.0
DEFAULT_WATER_MOLAR_MASS = 18.0
DEFAULT_WATER_DENSITY = 1000.0
TRESHOLD_SGAS = 1e-16
TRESHOLD_AMFG = 1e-16

PROPERTIES_USED_FOR_MASS_CALCULATION = ["RPORV", "SWAT", "DWAT", "BWAT", "SGAS", "DGAS",
                                        "BGAS", "AMFG", "YMFG", "XMF2", "YMF2"]
PROPERTIES_USED_FOR_VOLUME_CALCULATION = ["RPORV", "SGAS", "AMFG", "YMFG", "XMF2", "YMF2",
                                          "PORV", "DGAS", "BGAS", "DWAT", "BWAT"]

PROPERTIES_USED_FOR_CALCULATIONS = ["RPORV","PORV","SGAS","DGAS", "BGAS", "DWAT",
                                    "BWAT", "AMFG", "YMFG", "XMF2", "YMF2"]

class CalculationType(Enum):
    volume_extent = 0
    volume_actual = 1
    volume_actual_simple = 2
    mass = 3

@dataclass
class SourceData:
    x: np.ndarray
    y: np.ndarray
    DATES: List[str]
    VOL: Optional[List[np.ndarray]] = None
    SWAT: Optional[List[np.ndarray]] = None
    SGAS: Optional[List[np.ndarray]] = None
    RPORV: Optional[List[np.ndarray]] = None
    PORV: Optional[List[np.ndarray]] = None
    AMFG: Optional[List[np.ndarray]] = None
    YMFG: Optional[List[np.ndarray]] = None
    XMF2: Optional[List[np.ndarray]] = None
    YMF2: Optional[List[np.ndarray]] = None
    DWAT: Optional[List[np.ndarray]] = None
    DGAS: Optional[List[np.ndarray]] = None
    BWAT: Optional[List[np.ndarray]] = None
    BGAS: Optional[List[np.ndarray]] = None
    zone: Optional[np.ndarray] = None

@dataclass
class Co2DataAtTimeStep:
    date: str
    aqu_phase: np.ndarray
    gas_phase: np.ndarray
    volume_coverage: Optional[np.ndarray] = None

    def total_amount(self) -> np.ndarray:
        return self.aqu_phase + self.gas_phase

@dataclass
class Co2Data:
    x: np.ndarray
    y: np.ndarray
    data_list: List[Co2DataAtTimeStep]
    units: Literal["kg","m3"]
    zone: Optional[np.ndarray] = None

def _try_prop(unrst: EclFile,
              prop_name: str):
    try:
        prop = unrst[prop_name]
    except KeyError:
        prop = None
    return prop

def _read_props(
        unrst: EclFile,
        prop_names: List,
) -> List[np.ndarray]:
    props_att = {p: _try_prop(unrst, p) for p in prop_names}
    act_prop_names = [k for k in prop_names if props_att[k] is not None]
    act_props = {k: props_att[k] for k in act_prop_names}
    return act_props

def _fetch_properties(
        unrst: EclFile,
        prop_names: List
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    dates = [d.strftime("%Y%m%d") for d in unrst.report_dates]
    properties = _read_props(unrst, prop_names)
    properties = {p: {d[1]: properties[p][d[0]].numpy_copy()
                      for d in enumerate(dates)}
                  for p in properties}
    return properties, dates

def _identify_gas_less_cells(
        sgases: dict,
        amfgs: dict
) -> np.ndarray:
    gas_less = np.logical_and.reduce([np.abs(sgases[s]) < TRESHOLD_SGAS for s in sgases])
    gas_less &= np.logical_and.reduce([np.abs(amfgs[a]) < TRESHOLD_AMFG for a in amfgs])
    return gas_less

def _reduce_properties(properties: List,
                       keep_idx: np.ndarray):
    return {p: {d: properties[p][d][keep_idx] for d in properties[p]} for p in properties}

def _is_subset(first: List[str], second: List[str]) -> bool:
    return all(x in second for x in first)

def _extract_source_data(
        grid_file: str,
        unrst_file: str,
        props: List[str],
        init_file: Optional[str] = None,
        zone_file: Optional[str] = None
) -> SourceData:
    print("Start extracting source data")
    grid = EclGrid(grid_file)
    unrst = EclFile(unrst_file)
    init = EclFile(init_file)
    properties, dates = _fetch_properties(unrst, props)
    print("Done fetching properties")
    active = np.where(grid.export_actnum().numpy_copy() > 0)[0]
    print("Number of active grid cells: " + str(len(active)))
    if _is_subset(['SGAS', 'AMFG'], properties):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["AMFG"])
    elif _is_subset(['SGAS', 'XMF2'], properties):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["XMF2"])
    else:
        exit()
    global_active_idx = active[~gasless]
    properties = _reduce_properties(properties, ~gasless)
    xyz = [grid.get_xyz(global_index=a) for a in global_active_idx]  # Tuple with (x,y,z) for each cell
    print("Done xyz")
    cells_x = [coord[0] for coord in xyz]
    cells_y = [coord[1] for coord in xyz]
    zone = None
    if zone_file is not None:
        zone = xtgeo.gridproperty_from_file(zone_file, grid=grid)
        zone = zone.values.data[global_active_idx]
    VOL0 = [grid.cell_volume(global_index=x) for x in global_active_idx]
    properties['VOL'] = {d: VOL0 for d in dates}
    try:
        PORV = init["PORV"]
        properties['PORV'] = {d: PORV[0].numpy_copy()[global_active_idx] for d in dates}
    except KeyError:
        pass
    sd = SourceData(
        cells_x,
        cells_y,
        dates,
        **{
            p: v for p, v in properties.items()
        },
        **{'zone': zone}
    )
    return sd

def _mole_to_mass_fraction(x: np.ndarray,
                           m_co2: float,
                           m_h20: float) -> np.ndarray:
    return x * m_co2 / (m_h20 + (m_co2 - m_h20) * x)

def _set_calc_type_from_input_string(calc_type_input: str) -> CalculationType:
    if calc_type_input not in CalculationType._member_names_:
        print("Illegal calculation type: " + calc_type_input)
        print("Valid options:")
        for x in CalculationType._member_names_:
            print("  * " + x)
        print("Exiting")
        exit()
    return CalculationType[calc_type_input]

def _pflotran_co2mass(source_data: SourceData,
                      co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
                      water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS) -> Dict:
    dates = source_data.DATES
    dwat = source_data.DWAT
    dgas = source_data.DGAS
    amfg = source_data.AMFG
    ymfg = source_data.YMFG
    sgas = source_data.SGAS
    swat = source_data.SWAT
    eff_vols = source_data.PORV
    co2_mass = {}
    for t in dates:
        co2_mass[t] = [
            eff_vols[t] * (1-sgas[t]) * dwat[t] * _mole_to_mass_fraction(amfg[t], co2_molar_mass, water_molar_mass),
            eff_vols[t] * sgas[t] * dgas[t] * _mole_to_mass_fraction(ymfg[t], co2_molar_mass, water_molar_mass)
            ]
    return co2_mass

def _eclipse_co2mass(source_data: SourceData,
                     co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS) -> Dict:
    dates = source_data.DATES
    bgas = source_data.BGAS
    bwat = source_data.BWAT
    xmf2 = source_data.XMF2
    ymf2 = source_data.YMF2
    sgas = source_data.SGAS
    swat = source_data.SWAT
    eff_vols = source_data.RPORV
    conv_fact = co2_molar_mass
    co2_mass = {}
    for t in dates:
        co2_mass[t] = [conv_fact * bwat[t] * xmf2[t] * (1-sgas[t]) * eff_vols[t],
                       conv_fact * bgas[t] * ymf2[t] * sgas[t] * eff_vols[t]
                       ]
    return co2_mass

def _pflotran_co2_molar_volume(source_data: SourceData,
                               water_density: float,
                               co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
                               water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS) -> Dict:
    dates = source_data.DATES
    dgas = source_data.DGAS
    dwat = source_data.DWAT
    ymfg = source_data.YMFG
    amfg = source_data.AMFG
    co2_molar_vol = {}
    for t in dates:
        co2_molar_vol[t] = [(1 / amfg[t]) * (-water_molar_mass * (1 - amfg[t]) / (1000 * water_density) +
                                             (co2_molar_mass * amfg[t] + water_molar_mass * (1 - amfg[t])) / (
                                                         1000 * dwat[t])),
                            (1 / ymfg[t]) * (-water_molar_mass * (1 - ymfg[t]) / (1000 * water_density) +
                                             (co2_molar_mass * ymfg[t] + water_molar_mass * (1 - ymfg[t])) / (
                                                         1000 * dgas[t]))
                            ]
        co2_molar_vol[t][0] = [0 if x < 0 or y == 0 else x for x, y in zip(co2_molar_vol[t][0], amfg[t])]
        co2_molar_vol[t][1] = [0 if x < 0 or y == 0 else x for x, y in zip(co2_molar_vol[t][1], ymfg[t])]
    return co2_molar_vol

def _eclipse_co2_molar_volume(source_data, water_density: float = DEFAULT_WATER_DENSITY,
                              water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS) -> Dict:
    dates = source_data.DATES
    bgas = source_data.BGAS
    bwat = source_data.BWAT
    xmf2 = source_data.XMF2
    ymf2 = source_data.YMF2
    co2_molar_vol = {}
    for t in dates:
        co2_molar_vol[t] = [
            (1 / xmf2[t]) * (-water_molar_mass * (1 - xmf2[t]) / (1000 * water_density) + 1 / (1000 * bwat[t])),
            (1 / ymf2[t]) * (-water_molar_mass * (1 - ymf2[t]) / (1000 * water_density) + 1 / (1000 * bgas[t]))
            ]
        co2_molar_vol[t][0] = [0 if x < 0 or y == 0 else x for x, y in zip(co2_molar_vol[t][0], xmf2[t])]
        co2_molar_vol[t][1] = [0 if x < 0 or y == 0 else x for x, y in zip(co2_molar_vol[t][1], ymf2[t])]
    return co2_molar_vol

def _pflotran_co2_simple_volume(source_data: SourceData) -> Dict:
    dates = source_data.DATES
    sgas = source_data.SGAS
    ymfg = source_data.YMFG
    amfg = source_data.AMFG
    eff_vols = source_data.PORV
    co2_vol_st1 = {}
    for t in dates:
        co2_vol_st1[t] = [eff_vols[t] * (1 - sgas[t]) * amfg[t], eff_vols[t] * sgas[t] * ymfg[t]]
    return co2_vol_st1

def _eclipse_co2_simple_volume(source_data: SourceData) -> Dict:
    dates = source_data.DATES
    sgas = source_data.SGAS
    xmf2 = source_data.XMF2
    ymf2 = source_data.YMF2
    eff_vols = source_data.RPORV
    co2_vol_st1 = {}
    for t in dates:
        co2_vol_st1[t] = [eff_vols[t] * (1 - sgas[t]) * xmf2[t], eff_vols[t] * sgas[t] * ymf2[t]]
    return co2_vol_st1

def _calculate_co2_from_source_data(
        source_data: SourceData,
        calc_type: CalculationType,
        co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
        water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS,
        ) -> Co2Data:
    props_check = [x.name for x in fields(source_data) if x.name not in ['x', 'y', 'DATES', 'zone', 'VOL']]
    active_props_idx = np.where([getattr(source_data, x) is not None for x in props_check])[0]
    active_props = [props_check[i] for i in active_props_idx]
    if _is_subset(['SGAS'],active_props):
        if _is_subset(['PORV','RPORV'],active_props):
            active_props.remove('PORV')
        if _is_subset(['PORV','DGAS','DWAT','AMFG','YMFG'],active_props):
            source = 'PFlotran'
        else:
            if _is_subset(['RPORV','BGAS','BWAT','XMF2','YMF2'],active_props):
                source = 'Eclipse'
            else:
                print('Information is not enough to compute CO2 mass')
    else:
        print('Information is not enough to compute CO2 mass')
    if calc_type == CalculationType.volume_actual or calc_type == CalculationType.mass:
        if source == 'PFlotran':
            co2_mass_cell = _pflotran_co2mass(source_data, co2_molar_mass, water_molar_mass)
        else:
            co2_mass_cell = _eclipse_co2mass(source_data, co2_molar_mass)
        co2_mass_output = Co2Data(
            source_data.x,
            source_data.y,
            [Co2DataAtTimeStep(
                t,
                co2_mass_cell[t][0],
                co2_mass_cell[t][1],
                None) for t in co2_mass_cell],
            "kg",
            source_data.zone
        )
        if calc_type != CalculationType.mass:
          if source == 'PFlotran':
            water_density = np.array([x[1] if 1 - (source_data.AMFG[source_data.DATES[0]][x[0]]) == 1
                                      else np.mean(source_data.DWAT[source_data.DATES[0]][
                                        np.where(
                                          [y == 0 for y in
                                           source_data.AMFG[source_data.DATES[0]]])[
                                          0]])
                                      for x in enumerate(source_data.DWAT[source_data.DATES[0]])])
            molar_vols_co2 = _pflotran_co2_molar_volume(source_data, water_density, co2_molar_mass,
                                                        water_molar_mass)
          else:
              water_density = np.array(
                [water_molar_mass * x[1] if 1 - (source_data.XMF2[source_data.DATES[0]][x[0]]) == 1
                 else np.mean(source_data.BWAT[source_data.DATES[0]][
                   np.where(
                     [y == 0 for y in source_data.XMF2[source_data.DATES[0]]])[
                     0]])
                 for x in enumerate(source_data.BWAT[source_data.DATES[0]])])
              molar_vols_co2 = _eclipse_co2_molar_volume(source_data, water_density, water_molar_mass)
          co2_mass = {co2_mass_output.data_list[t].date: [co2_mass_output.data_list[t].aqu_phase,
                                                              co2_mass_output.data_list[t].gas_phase]
                          for t in range(0, len(co2_mass_data.data_list))}
          vols_co2 = {t: [a * b / (co2_molar_mass / 1000) for a, b in zip(molar_vols_co2[t], co2_mass[t])] for t
                          in
                          co2_mass}
          co2_amount = Co2Data(
                source_data.x,
                source_data.y,
                [Co2DataAtTimeStep(
                  t,
                  np.array(vols_co2[t][0]),
                  np.array(vols_co2[t][1]),
                  None) for t in vols_co2],
                "m3",
                source_data.zone
              )
        else:
            co2_amount = co2_mass_output
    else:
        if calc_type == CalculationType.volume_extent:
          props_idx = np.where([getattr(source_data, x) is not None for x in props_check])[0]
          props_names = [props_check[i] for i in props_idx]
          plume_props_names = [x for x in props_names if x in ['SGAS', 'AMFG', 'XMF2']]
          properties = {x: getattr(source_data, x) for x in plume_props_names}
          inactive_gas_cells = {x: _identify_gas_less_cells({x: properties[plume_props_names[0]][x]},
                                                              {x: properties[plume_props_names[1]][x]})
                                  for x in source_data.DATES}
          vols_ext = {t: np.array([0] * len(source_data.VOL[t])) for t in source_data.DATES}
          for t in source_data.DATES:
            vols_ext[t][~inactive_gas_cells[t]] = np.array(source_data.VOL[t])[~inactive_gas_cells[t]]
          co2_amount = Co2Data(
                source_data.x,
                source_data.y,
                [Co2DataAtTimeStep(
                    t,
                    None,
                    None,
                    np.array(vols_ext[t])
                ) for t in vols_ext],
                "m3",
                source_data.zone
            )
        else:
            if source == 'PFlotran':
                vols_co2 = _pflotran_co2_simple_volume(source_data)
            else:
                vols_co2 = _eclipse_co2_simple_volume(source_data)
            vols_co2_simp = {t: [vols_co2[t][0], vols_co2[t][1]] for t in vols_co2}
            co2_amount = Co2Data(source_data.x,
                                         source_data.y,
                                         [Co2DataAtTimeStep(
                                             t,
                                             np.array(vols_co2_simp[t][0]),
                                             np.array(vols_co2_simp[t][1]),
                                             None) for t in vols_co2_simp],
                                         "m3",
                                         source_data.zone)
    return co2_amount

def calculate_co2(
        grid_file: str,
        unrst_file: str,
        calc_type_input: Optional[str] = None,
        init_file: Optional[str] = None,
        zone_file: Optional[str] = None
) -> Co2Data:
    source_data = _extract_source_data(
            grid_file,unrst_file,PROPERTIES_USED_FOR_CALCULATIONS,init_file,zone_file)
    calc_type = _set_calc_type_from_input_string(calc_type_input.lower())
    co2_data = _calculate_co2_from_source_data(source_data, calc_type=calc_type)
    return co2_data


def main(arguments):
    # Not implemented (yet)
    # Use calculate_co2_mass() or calculate_co2_volume() directly
    pass

if __name__ == '__main__':
    main(sys.argv[1:])
