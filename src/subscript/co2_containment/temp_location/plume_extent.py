#!/usr/bin/env python

import sys
import argparse
import pandas as pd
import numpy as np
import csv

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
    g = EclGrid("{}.EGRID".format(case))
    rs = EclFile("{}.UNRST".format(case))

    # First calculate distance from injection point to center of all cells
    nactive = g.get_num_active()
    dist = np.zeros(shape=(nactive,))
    for i in range(nactive):
        center = g.get_xyz(active_index=i)
        dist[i] = np.sqrt( (center[0]-injxy[0])**2 + (center[1]-injxy[1])**2 )

    sgas_results = calc_plume_extent("SGAS", threshold_sgas, rs, dist)
    print(sgas_results)
    amfg_results = calc_plume_extent("AMFG", threshold_amfg, rs, dist)
    print(amfg_results)

    return (sgas_results, amfg_results)


def calc_plume_extent(rskey, threshold, rs, dist):
    # Find max plume distance for each step
    nsteps = len(rs.report_steps)
    dist_vs_date = np.zeros(shape=(nsteps,))
    for i in range(nsteps):
        data = rs[rskey][i].numpy_view()
        plumeix = np.where(data > threshold)[0]
        maxdist = 0.0
        if len(plumeix) > 0:
            maxdist = dist[plumeix].max()

        dist_vs_date[i] = maxdist

    output = []
    for i, d in enumerate(rs.report_dates):
        temp = [d.strftime('%Y-%m-%d'), dist_vs_date[i]]
        output.append(temp)

    return output


def main():
    args = make_parser().parse_args()
    case = args.case
    injxy = (args.injx, args.injy)
    threshold_sgas = args.threshold_sgas
    threshold_amfg = args.threshold_amfg

    (sgas_results, amfg_results) = calc_plume_extents(
        case,
        injxy,
        threshold_sgas,
        threshold_amfg,
    )
    
    # Convert into Pandas DataFrames
    sgas_df = pd.DataFrame.from_records(sgas_results, columns=["DATE", "MAX_DISTANCE_SGAS"])
    amfg_df = pd.DataFrame.from_records(amfg_results, columns=["DATE", "MAX_DISTANCE_AMFG"])

    # Merge them together
    df = pd.merge(sgas_df, amfg_df, on="DATE")

    # Export to CSV
    df.to_csv("share/results/tables/plumeextent.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
