#!/usr/bin/env python

import sys
import argparse
import pandas as pd
import numpy as np

from ecl.eclfile import EclFile
from ecl.grid import EclGrid

DEFAULT_THRESHOLD_SGAS = 0.2
DEFAULT_THRESHOLD_AMFG = 0.0005


def make_parser():
    parser = argparse.ArgumentParser(description="Calculate plume extent (distance)")
    parser.add_argument("case", help="Name of Eclipse case")
    parser.add_argument("--injx", default=560593, type=float, help="X-coordinate of injection point")
    parser.add_argument("--injy", default=6703786, type=float, help="Y-coordinate of injection point")
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
    case,
    injxy,
    threshold_sgas=DEFAULT_THRESHOLD_SGAS,
    threshold_amfg=DEFAULT_THRESHOLD_AMFG,
):
    grid = EclGrid("{}.EGRID".format(case))
    unrst = EclFile("{}.UNRST".format(case))

    # First calculate distance from injection point to center of all cells
    nactive = grid.get_num_active()
    dist = np.zeros(shape=(nactive,))
    for i in range(nactive):
        center = grid.get_xyz(active_index=i)
        dist[i] = np.sqrt( (center[0]-injxy[0])**2 + (center[1]-injxy[1])**2 )

    sgas_results = find_max_distances_per_time_step("SGAS", threshold_sgas, unrst, dist)
    print(sgas_results)
    amfg_results = find_max_distances_per_time_step("AMFG", threshold_amfg, unrst, dist)
    print(amfg_results)

    return (sgas_results, amfg_results)


def find_max_distances_per_time_step(attribute_key, threshold, unrst, dist):
    # Find max plume distance for each step
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
        temp = [d.strftime('%Y-%m-%d'), dist_vs_date[i]]
        output.append(temp)

    return output


def export_to_csv(sgas_results, amfg_results):
    # Convert into Pandas DataFrames
    sgas_df = pd.DataFrame.from_records(sgas_results, columns=["DATE", "MAX_DISTANCE_SGAS"])
    amfg_df = pd.DataFrame.from_records(amfg_results, columns=["DATE", "MAX_DISTANCE_AMFG"])

    # Merge them together
    df = pd.merge(sgas_df, amfg_df, on="DATE")

    # Export to CSV
    df.to_csv("share/results/tables/plumeextent.csv", index=False)


def main():
    args = make_parser().parse_args()

    (sgas_results, amfg_results) = calc_plume_extents(
        args.case,
        (args.injx, args.injy),
        args.threshold_sgas,
        args.threshold_amfg,
    )

    export_to_csv(sgas_results, amfg_results)

    return 0


if __name__ == "__main__":
    sys.exit(main())
