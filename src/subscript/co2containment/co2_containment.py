import argparse
import dataclasses
import pathlib
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import shapely.geometry

from .co2_calculation import (
    calculate_co2,
    Co2Data,
    CalculationType,
    _set_calc_type_from_input_string,
)

from .calculate import (
    calculate_co2_containment,
    ContainedCo2,
)


def calculate_out_of_bounds_co2(
    grid_file: str,
    unrst_file: str,
    init_file: str,
    compact: bool,
    calc_type_input: str,
    file_containment_polygon: Optional[str] = None,
    file_hazardous_polygon: Optional[str] = None,
    zone_file: Optional[str] = None
) -> pd.DataFrame:
    co2_data = calculate_co2(grid_file,
                             unrst_file,
                             calc_type_input,
                             init_file,
                             zone_file)
    print("Done with CO2 volume calculations")
    if file_containment_polygon is not None:
        containment_polygon = _read_polygon(file_containment_polygon)
    else:
        containment_polygon = None
    if file_hazardous_polygon is not None:
        hazardous_polygon = _read_polygon(file_hazardous_polygon)
    else:
        hazardous_polygon = None
    return calculate_from_co2_data(co2_data,
                                   containment_polygon,
                                   hazardous_polygon,
                                   compact,
                                   calc_type_input)


def calculate_from_co2_data(
    co2_data: Co2Data,
    containment_polygon: shapely.geometry.Polygon,
    hazardous_polygon: Union[shapely.geometry.Polygon, None],
    compact: bool,
    calc_type_input: str
) -> Union[pd.DataFrame, Dict[str, pd.DataFrame]]:
    calc_type = _set_calc_type_from_input_string(calc_type_input.lower())
    contained_co2 = calculate_co2_containment(
        co2_data,
        containment_polygon,
        hazardous_polygon,
        calc_type=calc_type
    )
    df = _construct_containment_table(contained_co2)
    if compact:
        return df
    if co2_data.zone is None:
        return _merge_date_rows(df, co2_data.units, calc_type)
    return {
        z: _merge_date_rows(g, co2_data.units, calc_type)
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


def _merge_date_rows(df: pd.DataFrame,
                     units: str,  # NBNB-AS
                     calc_type: CalculationType) -> pd.DataFrame:
    print("\nMerging data rows for data frame:")
    print(df)
    print("")
    df = df.drop("zone", axis=1)
    # Total
    df1 = (
        df
        .drop(["phase", "location"], axis=1)
        .groupby(["date"])
        .sum()
        .rename(columns={"amount": "total"})
    )
    total_df = df1.copy()
    if calc_type == CalculationType.volume_extent:
        df2 = (
            df
            .drop("phase", axis=1)
            .groupby(["location", "date"])
            .sum()
        )
        df2a = df2.loc[("contained",)].rename(
            columns={"amount": "total_contained"})
        df2b = df2.loc[("outside",)].rename(columns={"amount": "total_outside"})
        df2c = df2.loc[("hazardous",)].rename(
            columns={"amount": "total_hazardous"})
        for _df in [df2a, df2b, df2c]:
            total_df = total_df.merge(_df, on="date", how="left")
    else:
        df2 = (
            df
            .drop("location", axis=1)
            .groupby(["phase", "date"])
            .sum()
        )
        df2a = df2.loc["gas"].rename(columns={"amount": "total_gas"})
        df2b = df2.loc["aqueous"].rename(columns={"amount": "total_aqueous"})
        # Total by containment
        df3 = (
            df
            .drop("phase", axis=1)
            .groupby(["location", "date"])
            .sum()
        )
        df3a = df3.loc[("contained",)].rename(columns={"amount": "total_contained"})
        df3b = df3.loc[("outside",)].rename(columns={"amount": "total_outside"})
        df3c = df3.loc[("hazardous",)].rename(columns={"amount": "total_hazardous"})
        # Total by containment and phase
        df4 = (
            df
            .groupby(["phase", "location", "date"])
            .sum()
        )
        df4a = df4.loc["gas", "contained"].rename(columns={"amount": "gas_contained"})
        df4b = df4.loc["aqueous", "contained"].rename(columns={"amount": "aqueous_contained"})
        df4c = df4.loc["gas", "outside"].rename(columns={"amount": "gas_outside"})
        df4d = df4.loc["aqueous", "outside"].rename(columns={"amount": "aqueous_outside"})
        df4e = df4.loc["gas", "hazardous"].rename(columns={"amount": "gas_hazardous"})
        df4f = df4.loc["aqueous", "hazardous"].rename(columns={"amount": "aqueous_hazardous"})
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
    parser.add_argument("--calc_type_input", help="CO2 calculation options: mass / volume_extent / volume_actual / volume_actual_simple", default=None)
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
        arguments.compact,
        arguments.calc_type_input,
        arguments.containment_polygon,
        arguments.hazardous_polygon,
        arguments.zonefile
    )
    if isinstance(df, dict):
        of = pathlib.Path(arguments.outfile)
        [
            _df.to_csv(of.with_name(f"{of.stem}_{z}{of.suffix}"), index=False)
            for z, _df in df.items()
        ]
    else:
        df.to_csv(arguments.outfile, index=False)


if __name__ == "__main__":
    main(sys.argv[1:])
