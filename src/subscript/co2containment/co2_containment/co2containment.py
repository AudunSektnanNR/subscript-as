import argparse
import dataclasses
import pathlib
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shapely.geometry
from subscript.co2containment.co2_mass_calculation.co2_mass_calculation import calculate_co2_mass
from subscript.co2containment.co2_mass_calculation.co2_mass_calculation import Co2MassData

from .calculate import (
    calculate_co2_containment,
    ContainedCo2,
)


def calculate_out_of_bounds_co2(
    grid_file: str,
    unrst_file: str,
    init_file: str,
    file_containment_polygon: str,
    file_hazardous_polygon: str,
    compact: bool,
    zone_file: Optional[str] = None,
) -> pd.DataFrame:
    co2_mass_data = calculate_co2_mass(grid_file,
                                       unrst_file,
                                       init_file,
                                       zone_file)
    containment_polygon = _read_polygon(file_containment_polygon)
    hazardous_polygon = _read_polygon(file_hazardous_polygon)
    return calculate_from_co2_mass_data(co2_mass_data,
                                        containment_polygon,
                                        hazardous_polygon,
                                        compact)

def calculate_from_co2_mass_data(
    co2_mass_data: Co2MassData,
    containment_polygon: shapely.geometry.Polygon,
    hazardous_polygon: shapely.geometry.Polygon,
    compact: bool,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    contained_mass = calculate_co2_containment(
        co2_mass_data, containment_polygon, hazardous_polygon
    )
    df = _construct_containment_table(contained_mass)
    if compact:
        return df
    if co2_mass_data.zone is None:
        return _merge_date_rows(df)
    return {
        z: _merge_date_rows(g)
        for z, g in df.groupby("zone")
    }


def _read_polygon(polygon_file: str) -> shapely.geometry.Polygon:
    poly_xy = np.genfromtxt(polygon_file, skip_header=1, delimiter=",")[:, :2]
    return shapely.geometry.Polygon(poly_xy)


def _construct_containment_table(
    contained_co2: List[ContainedCo2],
) -> pd.DataFrame:
    records = [
        dataclasses.asdict(c)
        for c in contained_co2
    ]
    return pd.DataFrame.from_records(records)


def _merge_date_rows(df: pd.DataFrame) -> pd.DataFrame:
    print("")
    print(df)
    print("")
    df = df.drop("zone", axis=1)
    # print(df)
    # print("")
    # Total
    akg = "amount_kg"
    df1 = (
        df
        .drop(["phase", "inside_containment_boundary", "outside_boundaries", "inside_hazardous_boundary"], axis=1)
        .groupby(["date"])
        .sum()
        .rename(columns={akg: "total"})
    )
    # print(df1)
    # print("")
    # Total by phase
    df2 = (
        df
        .drop("inside_containment_boundary", axis=1)
        .groupby(["phase", "date"])
        .sum()
    )
    df2a = df2.loc["gas"].rename(columns={akg: "total_gas"})
    df2b = df2.loc["aqueous"].rename(columns={akg: "total_aqueous"})
    # Total by containment
    df3 = (
        df
        .drop("phase", axis=1)
        .groupby(["inside_containment_boundary", "date"])
        .sum()
    )
    df3a = df3.loc[(True,)].rename(columns={akg: "total_inside"})
    df3b = df3.loc[(False,)].rename(columns={akg: "total_outside"})
    # Total by containment and phase
    df4 = (
        df
        .groupby(["phase", "inside_containment_boundary", "date"])
        .sum()
    )
    df4a = df4.loc["gas", True].rename(columns={akg: "gas_inside"})
    df4b = df4.loc["aqueous", True].rename(columns={akg: "aqueous_inside"})
    df4c = df4.loc["gas", False].rename(columns={akg: "gas_outside"})
    df4d = df4.loc["aqueous", False].rename(columns={akg: "aqueous_outside"})
    # Merge data frames and append normalized values
    total_df = df1.copy()
    for _df in [df2a, df2b, df3a, df3b, df4a, df4b, df4c, df4d]:
        total_df = total_df.merge(_df, on="date", how="left")
    return total_df.reset_index()


def make_parser():
    pn = pathlib.Path(__file__).name
    parser = argparse.ArgumentParser(pn)
    parser.add_argument("grid", help="Grid (.EGRID) from which maps are generated")
    parser.add_argument("containment_polygon", help="Polygon that determines the bounds of the containment area")
    parser.add_argument("hazardous_polygon", help="Polygon that determines the bounds of the hazardous area")
    parser.add_argument("outfile", help="Output filename")
    parser.add_argument("--unrst", help="Path to UNRST file. Will assume same base name as grid if not provided", default=None)
    parser.add_argument("--init", help="Path to INIT file. Will assume same base name as grid if not provided", default=None)
    parser.add_argument("--zonefile", help="Path to file containing zone information", default=None)
    parser.add_argument("--compact", help="Write the output to a single file as compact as possible", action="store_true")
    return parser


def process_args(arguments):
    args = make_parser().parse_args(arguments)
    if args.unrst is None:
        args.unrst = args.grid.replace(".EGRID", ".UNRST")
    if args.init is None:
        args.init = args.grid.replace(".EGRID", ".INIT")
    return args


def main(arguments):
    arguments = process_args(arguments)
    df = calculate_out_of_bounds_co2(
        arguments.grid,
        arguments.unrst,
        arguments.init,
        arguments.containment_polygon,
        arguments.hazardous_polygon,
        arguments.compact,
        arguments.zonefile,
    )
    if isinstance(df, dict):
        of = pathlib.Path(arguments.outfile)
        [
            _df.to_csv(of.with_name(f"{of.stem}_{z}{of.suffix}"), index=False)
            for z, _df in df.items()
        ]
    else:
        df.to_csv(arguments.outfile, index=False)


if __name__ == '__main__':
    main(sys.argv[1:])
