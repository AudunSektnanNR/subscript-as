import argparse
import dataclasses
import pathlib
import sys
from typing import Dict, List, Optional, Union
from functools import reduce

import numpy as np
import pandas as pd
import shapely.geometry

from subscript.co2containment.co2_mass_calculation.co2_mass_calculation import calculate_co2_volume
from subscript.co2containment.co2_mass_calculation.co2_mass_calculation import Co2VolumeData

from subscript.co2containment.co2_containment.calculate import (
    calculate_co2_containment_vol,
    ContainedCo2Vol,
)

def calculate_out_of_bounds_co2(
    grid_file: str,
    unrst_file: str,
    init_file: str,
    file_containment_polygon: str,
    compact: bool,
    zone_file: Optional[str] = None,
    file_hazardous_polygon: Optional[str] = None,
) -> pd.DataFrame:
    co2_volume_data = calculate_co2_volume(grid_file,
                                       unrst_file,
				       threshold,
                                       zone_file)
    containment_polygon = _read_polygon(file_containment_polygon)
    if file_hazardous_polygon is not None:
        hazardous_polygon = _read_polygon(file_hazardous_polygon)
    else:
        hazardous_polygon = None
    return calculate_from_co2_volume_data(co2_volume_data,
                                        containment_polygon,
                                        hazardous_polygon,
                                        compact)

def calculate_from_co2_volume_data(
    co2_volume_data: Co2VolumeData,
    containment_polygon: shapely.geometry.Polygon,
    hazardous_polygon: Union[shapely.geometry.Polygon, None],
    compact: bool,
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    contained_volume = calculate_co2_containment_vol(
        co2_volume_data, containment_polygon, hazardous_polygon
    )
    df = _construct_containment_table_vol(contained_volume)
    if compact:
        return df
    if co2_volume_data.zone is None:
        return _merge_date_rows_vol(df)
    return {
        z: _merge_date_rows(g)
        for z, g in df.groupby("zone")
    }

def _read_polygon(polygon_file: str) -> shapely.geometry.Polygon:
    poly_xy = np.genfromtxt(polygon_file, skip_header=1, delimiter=",")[:, :2]
    return shapely.geometry.Polygon(poly_xy)

def _construct_containment_table_vol(
    contained_co2: List[ContainedCo2Vol],
) -> pd.DataFrame:
    records = {p:pd.DataFrame.from_records([
        dataclasses.asdict(c)
        for c in contained_co2[p]
	])
	for p in contained_co2
    }
    records = [records[p].rename(columns={'amount_m3':'amount_m3_'+p}) for p in records]
    return reduce(lambda left, right:     # Merge DataFrames in list
                     pd.merge(left , right,
                              on = ["date","phase","location","zone"],
                              how = "inner"),
                     records)

def _merge_date_rows_vol(df: pd.DataFrame) -> pd.DataFrame:
    print("")
    print(df)
    print("")
    df = df.drop("zone", axis=1)
    # print(df)
    # print("")
    # Total
    am3_SGAS = "amount_m3_SGAS"
    am3_AMFG = "amount_m3_AMFG"
    df1 = (
        df
	.drop(df[df["phase"]=="extent"].index)
        .drop(["phase", "location"], axis=1)
        .groupby(["date"])
        .sum()
        .rename(columns={am3_SGAS: "total_SGAS",am3_AMFG: "total_AMFG"})
    )
    print(df1)
    print("")
    # Total by phase
    df2 = (
        df
        .drop("location", axis=1)
        .groupby(["phase", "date"])
        .sum()
    )
    df2a = df2.loc["gas"].rename(columns={am3_SGAS: "total_gas_SGAS",am3_AMFG: "total_gas_AMFG"})
    df2b = df2.loc["aqueous"].rename(columns={am3_SGAS: "total_aqueous_SGAS",am3_AMFG: "total_aqueous_AMFG"})
    df2c = df2.loc["extent"].rename(columns={am3_SGAS: "total_extent_SGAS",am3_AMFG: "total_extent_AMFG"})
    # Total by containment
    df3 = (
        df
        .drop(df[df["phase"]=="extent"].index)
        .drop("phase", axis=1)
        .groupby(["location", "date"])
        .sum()
    )
    df3a = df3.loc[("contained",)].rename(columns={am3_SGAS: "total_contained_SGAS",am3_AMFG: "total_contained_AMFG"})
    df3b = df3.loc[("outside",)].rename(columns={am3_SGAS: "total_outside_SGAS",am3_AMFG: "total_outside_AMFG"})
    df3c = df3.loc[("hazardous",)].rename(columns={am3_SGAS: "total_hazardous_SGAS",am3_AMFG: "total_hazardous_AMFG"})
    print(df3a)
    # Total by containment and phase
    df4 = (
        df
        .groupby(["phase", "location", "date"])
        .sum()
    )
    df4a = df4.loc["gas", "contained"].rename(columns={am3_SGAS: "gas_contained_SGAS",am3_AMFG: "gas_contained_AMFG"})
    df4b = df4.loc["aqueous", "contained"].rename(columns={am3_SGAS: "aqueous_contained_SGAS",am3_AMFG: "aqueous_contained_AMFG"})
    df4c = df4.loc["gas", "outside"].rename(columns={am3_SGAS: "gas_outside_SGAS",am3_AMFG: "gas_outside_AMFG"})
    df4d = df4.loc["aqueous", "outside"].rename(columns={am3_SGAS: "aqueous_outside_SGAS",am3_AMFG: "aqueous_outside_AMFG"})
    df4e = df4.loc["gas", "hazardous"].rename(columns={am3_SGAS: "gas_hazardous_SGAS",am3_AMFG: "gas_hazardous_AMFG"})
    df4f = df4.loc["aqueous", "hazardous"].rename(columns={am3_SGAS: "aqueous_hazardous_SGAS",am3_AMFG: "aqueous_hazardous_AMFG"})
    df4g = df4.loc["extent", "contained"].rename(columns={am3_SGAS: "extent_contained_SGAS",am3_AMFG: "extent_contained_AMFG"})
    df4h = df4.loc["extent", "outside"].rename(columns={am3_SGAS: "extent_outside_SGAS",am3_AMFG: "extent_outside_AMFG"})
    df4i = df4.loc["extent", "hazardous"].rename(columns={am3_SGAS: "extent_hazardous_SGAS",am3_AMFG: "extent_hazardous_AMFG"})
    print(df4a)
    # Merge data frames and append normalized values
    total_df = df1.copy()
    for _df in [df2a, df2b,df2c, df3a, df3b, df3c, df4a, df4b, df4c, df4d, df4e, df4f, df4g, df4h, df4i]:
        total_df = total_df.merge(_df, on="date", how="left")
    return total_df.reset_index()


def make_parser():
    pn = pathlib.Path(__file__).name
    parser = argparse.ArgumentParser(pn)
    parser.add_argument("grid", help="Grid (.EGRID) from which maps are generated")
    parser.add_argument("containment_polygon", help="Polygon that determines the bounds of the containment area")
    parser.add_argument("outfile", help="Output filename")
    parser.add_argument("--unrst", help="Path to UNRST file. Will assume same base name as grid if not provided", default=None)
    parser.add_argument("--init", help="Path to INIT file. Will assume same base name as grid if not provided", default=None)
    parser.add_argument("--zonefile", help="Path to file containing zone information", default=None)
    parser.add_argument("--compact", help="Write the output to a single file as compact as possible", action="store_true")
    parser.add_argument("--hazardous_polygon", help="Polygon that determines the bounds of the hazardous area", default=None)
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
        arguments.compact,
        arguments.zonefile,
        arguments.hazardous_polygon,
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
