import argparse
import dataclasses
import pathlib
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shapely.geometry

from subscript.co2containment.co2_calculation.co2_calculation import calculate_co2_volume
from subscript.co2containment.co2_calculation.co2_calculation import Co2VolumeData

from subscript.co2containment.co2_mass_containment.calculate import (
    calculate_co2_containment_vol,
    ContainedCo2Vol,
)

def calculate_out_of_bounds_co2(
    grid_file: str,
    unrst_file: str,
    init_file: str,
    file_containment_polygon: str,
    compact: bool,
    vol_type: str,
    zone_file: Optional[str] = None,
    file_hazardous_polygon: Optional[str] = None,
) -> pd.DataFrame:
    print("Calculate out of bounds CO2 for volume type: " + vol_type)
    co2_volume_data = calculate_co2_volume(grid_file,
                                           unrst_file,
				                           vol_type,
				                           init_file,
                                           zone_file)
    print("Done with CO2 volume calculations")
    containment_polygon = _read_polygon(file_containment_polygon)
    if file_hazardous_polygon is not None:
        hazardous_polygon = _read_polygon(file_hazardous_polygon)
    else:
        hazardous_polygon = None
    return calculate_from_co2_volume_data(co2_volume_data,
                                          containment_polygon,
                                          hazardous_polygon,
                                          compact,
					                      vol_type)

def calculate_from_co2_volume_data(
    co2_volume_data: Co2VolumeData,
    containment_polygon: shapely.geometry.Polygon,
    hazardous_polygon: Union[shapely.geometry.Polygon, None],
    compact: bool,
    vol_type: str
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    contained_volume = calculate_co2_containment_vol(
        co2_volume_data, containment_polygon, hazardous_polygon, vol_type
    )
    df = _construct_containment_table_vol(contained_volume)
    if compact:
        return df
    if co2_volume_data.zone is None:
        return _merge_date_rows_vol(df, vol_type)
    return {
        z: _merge_date_rows_vol(g, vol_type)
        for z, g in df.groupby("zone")
    }

def _read_polygon(polygon_file: str) -> shapely.geometry.Polygon:
    poly_xy = np.genfromtxt(polygon_file, skip_header=1, delimiter=",")[:, :2]
    return shapely.geometry.Polygon(poly_xy)

def _construct_containment_table_vol(
    contained_co2: List[ContainedCo2Vol],
) -> pd.DataFrame:
    records = pd.DataFrame.from_records([
        dataclasses.asdict(c)
        for c in contained_co2
	])
    return records

def _merge_date_rows_vol(df: pd.DataFrame,
                         vol_type: str) -> pd.DataFrame:
    print("\nMerging data rows for data frame:")
    print(df)
    print("")
    df = df.drop("zone", axis=1)
    # Total
    am3 = "amount_m3"
    if vol_type == 'Extent':
        df1 = (
            df
            .drop(["phase", "location"], axis=1)
            .groupby(["date"])
            .sum()
            .rename(columns={am3: "total"})
        )
        # Total by containment
        df2 = (
            df
            .drop("phase", axis=1)
            .groupby(["location", "date"])
            .sum()
        )
        df2a = df2.loc[("contained",)].rename(
            columns={am3: "total_contained"})
        df2b = df2.loc[("outside",)].rename(columns={am3: "total_outside"})
        df2c = df2.loc[("hazardous",)].rename(
            columns={am3: "total_hazardous"})
        total_df = df1.copy()
        for _df in [df2a, df2b, df2c]:
            total_df = total_df.merge(_df, on="date", how="left")
    elif vol_type == 'Actual' or vol_type == 'Actual_simple':
        df1 = (
            df
            .drop(["phase", "location"], axis=1)
            .groupby(["date"])
            .sum()
            .rename(columns={am3: "total"})
        )
        # Total by phase
        df2 = (
            df
            .drop("location", axis=1)
            .groupby(["phase", "date"])
            .sum()
        )
        df2a = df2.loc["gas"].rename(columns={am3: "total_gas"})
        df2b = df2.loc["aqueous"].rename(columns={am3: "total_aqueous"})
        # Total by containment
        df3 = (
            df
            .drop("phase", axis=1)
            .groupby(["location", "date"])
            .sum()
        )
        df3a = df3.loc[("contained",)].rename(columns={am3: "total_contained"})
        df3b = df3.loc[("outside",)].rename(columns={am3: "total_outside"})
        df3c = df3.loc[("hazardous",)].rename(columns={am3: "total_hazardous"})
        # Total by containment and phase
        df4 = (
            df
            .groupby(["phase", "location", "date"])
            .sum()
        )
        df4a = df4.loc["gas", "contained"].rename(columns={am3: "gas_contained"})
        df4b = df4.loc["aqueous", "contained"].rename(columns={am3: "aqueous_contained"})
        df4c = df4.loc["gas", "outside"].rename(columns={am3: "gas_outside"})
        df4d = df4.loc["aqueous", "outside"].rename(columns={am3: "aqueous_outside"})
        df4e = df4.loc["gas", "hazardous"].rename(columns={am3: "gas_hazardous"})
        df4f = df4.loc["aqueous", "hazardous"].rename(columns={am3: "aqueous_hazardous"})
        # Merge data frames and append normalized values
        total_df = df1.copy()
        for _df in [df2a, df2b, df3a, df3b, df3c, df4a, df4b, df4c, df4d, df4e, df4f]:
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
    parser.add_argument("--vol_type",help="Volumetric extent or actual CO2 volume",default="None")
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
	    arguments.vol_type,
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
