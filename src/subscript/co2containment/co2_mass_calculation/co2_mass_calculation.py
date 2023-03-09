from dataclasses import dataclass, fields
from typing import Dict, List, Optional, Tuple
import sys

import numpy as np
import xtgeo
from ecl.eclfile import EclFile
from ecl.grid import EclGrid

TRESHOLD_SGAS = 1e-16
TRESHOLD_AMFG = 1e-16
DEFAULT_CO2_MOLAR_MASS = 44.0
DEFAULT_WATER_MOLAR_MASS = 18.0


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
class Co2MassDataAtTimeStep:
    date: str
    gas_phase_kg: np.ndarray
    aqu_phase_kg: np.ndarray

    def total_weight(self) -> np.ndarray:
        return self.aqu_phase_kg + self.gas_phase_kg


@dataclass
class Co2MassData:
    x: np.ndarray
    y: np.ndarray
    data_list: List[Co2MassDataAtTimeStep]
    zone: Optional[np.ndarray] = None


@dataclass
class Co2VolumeDataAtTimeStep:
    date: str
    volume_coverage: np.ndarray  # Or volume_extent ?
    volume_actual_co2: np.ndarray


@dataclass
class Co2VolumeData:
    x: np.ndarray
    y: np.ndarray
    data_list: List[Co2VolumeDataAtTimeStep]
    zone: Optional[np.ndarray] = None

def _try_prop(unrst:EclFile,
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
    props_att = {p:_try_prop(unrst,p) for p in prop_names}
    act_prop_names = [k for k in prop_names if props_att[k] is not None]
    act_props = {k:props_att[k] for k in act_prop_names}
    return act_props

def _fetch_properties(
    unrst: EclFile,
    prop_names: List
) -> Tuple[Dict[str, List[np.ndarray]], List[str]]:
    dates = [d.strftime("%Y%m%d") for d in unrst.report_dates]
    properties = _read_props(unrst,prop_names)
    properties = {p:{d[1]: properties[p][d[0]].numpy_copy() 
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
    return {p:{d: properties[p][d][keep_idx] for d in properties[p]} for p in properties}

def _extract_source_data(
    grid_file: str,
    unrst_file: str,
    props: List[str],
    init_file: Optional[str] = None,
    zone_file: Optional[str] = None
) -> SourceData:
    print("_extract_source_data()")
    grid = EclGrid(grid_file)
    unrst = EclFile(unrst_file)
    init = EclFile(init_file)
    properties, dates = _fetch_properties(unrst,props)
    print("Done fetching properties")
    active = np.where(grid.export_actnum().numpy_copy() > 0)[0]
    print("Number of active grid cells: " + str(len(active)))
    if set(['SGAS','AMFG']).issubset(set([x for x in properties])):
        gasless = _identify_gas_less_cells(properties["SGAS"], properties["AMFG"])
    else:
        if set(['SGAS','YMFG2']).issubset(set([x for x in properties])):
            gasless = _identify_gas_less_cells(properties["SGAS"], properties["YMG2"])
        else:
            exit()

    global_active_idx = active[~gasless]
    properties = _reduce_properties(properties,~gasless)

    xyz = [grid.get_xyz(global_index=a) for a in global_active_idx] #Tuple with (x,y,z) for each cell
    print("Done xyz")
    cells_x = [coord[0] for coord in xyz]
    cells_y = [coord[1] for coord in xyz]
    zone = None
    if zone_file is not None:
        zone = xtgeo.gridproperty_from_file(zone_file, grid=grid)
        zone = zone.values.data[global_active_idx]
    properties['VOL'] = {d:[grid.cell_volume(global_index=x) for x in global_active_idx] for d in dates}
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


def _mole_to_mass_fraction(x, m_co2, m_h20):
    return x * m_co2 / (m_h20 + (m_co2 - m_h20) * x)

def cut_threshold(x, threshold):
    return np.where(x > threshold)

def _pflotran_co2mass(source_data,
                     co2_molar_mass=DEFAULT_CO2_MOLAR_MASS,
                     water_molar_mass=DEFAULT_WATER_MOLAR_MASS):
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
            eff_vols[t] * sgas[t] * dgas[t] * _mole_to_mass_fraction(ymfg[t], co2_molar_mass, water_molar_mass),
            eff_vols[t] * swat[t] * dwat[t] * _mole_to_mass_fraction(amfg[t], co2_molar_mass, water_molar_mass)]
    return co2_mass

def _eclipse_co2mass(source_data, co2_molar_mass=DEFAULT_CO2_MOLAR_MASS):
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
        co2_mass[t] = [conv_fact * bgas[t] * ymf2[t] * sgas[t] * eff_vols[t],
                       conv_fact * bwat[t] * xmf2[t] * swat[t] * eff_vols[t]]
    return co2_mass

def _calculate_co2_mass_from_source_data(
    source_data: SourceData,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS
) -> Co2MassData:
    print("_calculate_co2_mass_from_source_data()")
    props_check = list(set([field.name for field in fields(source_data)]).difference(set(['x','y','DATES','VOL','zone'])))
    active_props_idx = np.where([getattr(source_data, x) is not None for x in props_check])[0]
    active_props = [props_check[i] for i in active_props_idx]
    print("Available properties:")
    print(active_props)

    if set(['SGAS','SWAT']).issubset(set(active_props)):
        if set(['PORV','RPORV']).issubset(set(active_props)):
            active_props.remove('PORV')

        if set(['PORV','SGAS', 'SWAT', 'DGAS', 'DWAT', 'AMFG', 'YMFG']).issubset(set(active_props)):
            source = 'PFlotran'
            print('Data Source is ' + source)
        else:
            if set(['RPORV', 'SGAS', 'SWAT', 'BGAS', 'BWAT', 'XMF2', 'YMF2']).issubset(set(active_props)):
                source = 'Eclipse'
                print('Data Source is ' + source)
            else:
                print('Information is not enough to compute CO2 mass')
                exit()
    else:
        print('Information is not enough to compute CO2 mass')
        exit()

    if source == 'PFlotran':
        co2_mass_cell = _pflotran_co2mass(source_data,co2_molar_mass,water_molar_mass)
    else:
        co2_mass_cell = _eclipse_co2mass(source_data,co2_molar_mass)
    co2_mass_data = Co2MassData(
        source_data.x,
        source_data.y,
        [
            Co2MassDataAtTimeStep(
                x,
                co2_mass_cell[x][0],
                co2_mass_cell[x][1]
            )
            for x in co2_mass_cell
        ],
        source_data.zone
    )
    return co2_mass_data


def _calculate_co2_volume_from_source_data(
    source_data: SourceData,
    co2_molar_mass: float = DEFAULT_CO2_MOLAR_MASS,  # Not needed ?
    water_molar_mass: float = DEFAULT_WATER_MOLAR_MASS  # Not needed ?
) -> Co2VolumeData:
    # Similar to _calculate_co2_mass_from_source_data
    pass


def calculate_co2_mass(
    grid_file: str,
    unrst_file: str,
    init_file: Optional[str] = None,
    zone_file: Optional[str] = None
) -> Co2MassData:
    props = ["RPORV", "SWAT", "DWAT", "BWAT", "SGAS", "DGAS",
             "BGAS", "AMFG", "YMFG", "XMF2", "YMF2"]
    source_data = _extract_source_data(
        grid_file, unrst_file, props, init_file, zone_file
    )
    co2_mass_data = _calculate_co2_mass_from_source_data(source_data)
    print("done with co2_mass")
    return co2_mass_data


def calculate_co2_volume(
    grid_file: str,
    unrst_file: str,
    init_file: str,
    poro_keyword: str,
    zone_file: Optional[str] = None
) -> Co2VolumeData:
    source_data = _extract_source_data(
        grid_file, unrst_file, init_file, poro_keyword, zone_file
    )
    co2_volume_data = _calculate_co2_volume_from_source_data(source_data)
    return co2_volume_data


def main(arguments):
    # Not implemented (yet)
    # Use calculate_co2_mass() or calculate_co2_volume() directly
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
