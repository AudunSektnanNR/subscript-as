#!/usr/bin/env python


################################################################################
# Script calculating the area extent of the plume depending on which map / date are present in the share/results/maps folder
#
# Created by: Jorge Sicacha (NR), Oct 2022
# Modfied by: Floriane Mortier (fmmo), Nov 2022 - To fit FMU workflow
#
################################################################################

import sys
import os
import glob
import re
from datetime import datetime
import pandas as pd
import xtgeo
import numpy as np
import pathlib
from itertools import product
from pathos.multiprocessing import ProcessingPool as Pool
import plotly.graph_objects as go
import argparse


def make_parser():
    parser = argparse.ArgumentParser(description="Calculate plume area")
    parser.add_argument("input", help="Path to maps created through XTGeoapp")

    return parser


def find_formations(search_path, rskey):
    formation_list = []

    for file in glob.glob(search_path + "*max_" + rskey + "*.gri"):
        fm_name = pathlib.Path(file).stem.split("--")[0]

        if fm_name in formation_list:
            pass
        else:
            formation_list.append(fm_name)

    return formation_list


def find_dates(search_path, fm, rskey):
    date_list = []

    for file in glob.glob(search_path + fm[0] + "*max_" + rskey + "*.gri"):
        full_date = pathlib.Path(file).stem.split("--")[2]
        year = full_date[0:4]

        if year in date_list:
            pass
        else:
            date_list.append(year)

    return date_list


def neigh_nodes(x):  # If all the four nodes of the cell are not masked we count the area
    sq_vert = {(x[0] + 1, x[1]), (x[0], int(x[1]) + 1), (x[0] + 1, x[1] + 1)}

    return sq_vert


def calc_plume_area(rskey):
    args = make_parser().parse_args()
    path = args.input

    var = "max_" + rskey
    print("***" + rskey + "***")

    formation = np.array(find_formations(path, rskey))
    print("Formations found: ", formation)

    year = np.array(find_dates(path, formation, rskey))
    print("Dates found: ", year)

    area_array = (product(formation, var, year))

    dict_out = []
    for fm in formation:
        for y in year:
            path_file = glob.glob(path + fm + "--" + var + "--" + y + "*.gri")
            print("Path_file: ", path_file)
            # path_file = path + fm + "--" + var + "--" + y + "0101.gri"
            mysurf = xtgeo.surface_from_file(path_file[0])
            use_nodes = np.ma.nonzero(mysurf.values)  # Indexes of the existing nodes
            use_nodes = set(list(tuple(zip(use_nodes[0], use_nodes[1]))))
            all_neigh_nodes = list(map(neigh_nodes, use_nodes))
            test0 = [xx.issubset(use_nodes) for xx in all_neigh_nodes]
            dict_out_temp = [float(y), float(sum(t * mysurf.xinc * mysurf.yinc for t in test0)), fm]
            dict_out.append(dict_out_temp)

    return dict_out


def main():
    sgas_results = calc_plume_area("sgas")
    if sgas_results:
        print("Sgas areas sucessfully collected.")

    amfg_results = calc_plume_area("amfg")
    if amfg_results:
        print("Amfg areas sucessfully collected.")

    # Convert into Pandas DataFrames
    sgas_df = pd.DataFrame.from_records(sgas_results, columns=["DATE", "AREA_SGAS", "FORMATION_SGAS"])
    sgas_df = sgas_df.pivot(index="DATE", columns="FORMATION_SGAS", values="AREA_SGAS")
    sgas_df.reset_index(inplace=True)
    sgas_df.columns.name = None
    sgas_df.columns = [x + "_SGAS" if x != "DATE" else x for x in sgas_df.columns]
    amfg_df = pd.DataFrame.from_records(amfg_results, columns=["DATE", "AREA_AMFG", "FORMATION_AMFG"])
    amfg_df = amfg_df.pivot(index="DATE", columns="FORMATION_AMFG", values="AREA_AMFG")
    amfg_df.reset_index(inplace=True)
    amfg_df.columns.name = None
    amfg_df.columns = [x + "_AMFG" if x != "DATE" else x for x in amfg_df.columns]

    # Merge them together
    df = pd.merge(sgas_df, amfg_df)

    # Export to CSV
    df.to_csv("share/results/tables/plumearea.csv", index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
