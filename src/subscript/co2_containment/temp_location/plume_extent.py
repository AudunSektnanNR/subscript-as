#!/usr/bin/env python

import argparse
from pathlib import Path
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
from ecl.eclfile import EclFile
from ecl.grid import EclGrid

DEFAULT_THRESHOLD_SGAS = 0.2
DEFAULT_THRESHOLD_AMFG = 0.0005


def __make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Calculate plume extent (distance)")
    parser.add_argument("case", help="Name of Eclipse case")
    parser.add_argument(
        "--well_name", help="Name of injection well to calculate plume extent from",
        default=None,
    )
    parser.add_argument(
        "--x_coord", help="Value of x coordinate to calculate plume extent from. Can be used instead of --well_name."
    )
    parser.add_argument(
        "--y_coord", help="Value of y coordinate to calculate plume extent from. Can be used instead of --well_name."
    )
    parser.add_argument(
        "--output",
        help="Path to output CSV file",
        default="share/results/tables/plumeextent.csv",
    )
    parser.add_argument(
        "--threshold_sgas",
        default=DEFAULT_THRESHOLD_SGAS,
        type=float,
        help="Threshold for SGAS",
    )
    parser.add_argument(
        "--threshold_amfg",
        default=DEFAULT_THRESHOLD_AMFG,
        type=float,
        help="Threshold for AMFG",
    )

    return parser


def calc_plume_extents(
    case: str,
    injxy: Tuple[float, float],
    threshold_sgas: float = DEFAULT_THRESHOLD_SGAS,
    threshold_amfg: float = DEFAULT_THRESHOLD_AMFG,
) -> Tuple[List[List], List[List]]:
    """
    Find plume extents per date for SGAS and AMFG.
    """
    grid = EclGrid("{}.EGRID".format(case))
    unrst = EclFile("{}.UNRST".format(case))

    # First calculate distance from injection point to center of all cells
    nactive = grid.get_num_active()
    dist = np.zeros(shape=(nactive,))
    for i in range(nactive):
        center = grid.get_xyz(active_index=i)
        dist[i] = np.sqrt((center[0] - injxy[0]) ** 2 + (center[1] - injxy[1]) ** 2)

    sgas_results = __find_max_distances_per_time_step(
        "SGAS", threshold_sgas, unrst, dist
    )
    print(sgas_results)

    amfg_results = __find_max_distances_per_time_step(
        "AMFG", threshold_amfg, unrst, dist
    )
    print(amfg_results)

    return (sgas_results, amfg_results)


def __find_max_distances_per_time_step(
    attribute_key: str, threshold: float, unrst: EclFile, dist: np.ndarray
) -> List[List]:
    """
    Find max plume distance for each step
    """
    nsteps = len(unrst.report_steps)
    dist_vs_date = np.zeros(shape=(nsteps,))
    for i in range(nsteps):
        data = unrst[attribute_key][i].numpy_view()
        plumeix = np.where(data > threshold)[0]
        maxdist = 0.0
        if len(plumeix) > 0:
            maxdist = dist[plumeix].max()

        dist_vs_date[i] = maxdist

    output = []
    for i, d in enumerate(unrst.report_dates):
        temp = [d.strftime("%Y-%m-%d"), dist_vs_date[i]]
        output.append(temp)

    return output


def __export_to_csv(
    sgas_results: List[List], amfg_results: List[List], output_file: str
):
    # Convert into Pandas DataFrames
    sgas_df = pd.DataFrame.from_records(
        sgas_results, columns=["DATE", "MAX_DISTANCE_SGAS"]
    )
    amfg_df = pd.DataFrame.from_records(
        amfg_results, columns=["DATE", "MAX_DISTANCE_AMFG"]
    )

    # Merge them together
    df = pd.merge(sgas_df, amfg_df, on="DATE")

    # Export to CSV
    df.to_csv(output_file, index=False)


def __calculate_well_coordinates(case: str, well_name: str) -> Tuple[float, float]:
    """
    Find coordinates of injection point
    """
    p = Path(case).parents[2]
    p2 = p / "share" / "results" / "wells" / "well_picks.csv"

    df = pd.read_csv(p2)
    df = df[df["WELL"] == well_name]

    df = df[df["X_UTME"].notna()]
    df = df[df["Y_UTMN"].notna()]

    max_id = df["MD"].idxmax()
    max_md_row = df.loc[max_id]
    x = max_md_row["X_UTME"]
    y = max_md_row["Y_UTMN"]

    return (x, y)


def main():
    """
    Calculate plume extent using EGRID and UNRST-files. Calculated for SGAS
    and AMFG. Output is plume extent per date written to a CSV file.
    """
    args = __make_parser().parse_args()

    if args.x_coord and args.y_coord:
        injxy = (float(args.x_coord), float(args.y_coord))
    elif args.well_name:
        injxy = __calculate_well_coordinates(
            args.case,
            args.well_name,
        )
    else:
        print("Invalid input. Specify either --well_name or provide both --x_coord and --y_coord.")
        exit()

    (sgas_results, amfg_results) = calc_plume_extents(
        args.case,
        injxy,
        args.threshold_sgas,
        args.threshold_amfg,
    )

    __export_to_csv(sgas_results, amfg_results, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
