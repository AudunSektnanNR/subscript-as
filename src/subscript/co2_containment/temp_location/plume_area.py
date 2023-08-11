#!/usr/bin/env python


################################################################################
# Script calculating the area extent of the plume depending on which map / date are present in the share/results/maps folder
#
# Created by : Jorge Sicacha (NR), Oct 2022
# Modified by: Floriane Mortier (fmmo), Nov 2022 - To fit FMU workflow
#
################################################################################

import os
import sys
import glob
import pandas as pd
import xtgeo
import numpy as np
import pathlib
import argparse


def __make_parser():
    parser = argparse.ArgumentParser(description="Calculate plume area")
    parser.add_argument("input", help="Path to maps created through XTGeoapp")

    return parser


def __find_formations(search_path, rskey):
    # Try different capitalizations of rskey:
    file_names_1 = glob.glob(search_path + "*max_" + rskey + "*.gri")
    file_names_2 = glob.glob(search_path + "*max_" + rskey.lower() + "*.gri")
    file_names_3 = glob.glob(search_path + "*max_" + rskey.upper() + "*.gri")

    if file_names_1:
        rskey_updated = rskey
    elif file_names_2:
        rskey_updated = rskey.lower()
    elif file_names_3:
        rskey_updated = rskey.upper()
    else:
        text = f"No surface files found."
        raise FileNotFoundError(text)

    formation_list = []
    for file in glob.glob(search_path + "*max_" + rskey_updated + "*.gri"):
        fm_name = pathlib.Path(file).stem.split("--")[0]

        if fm_name in formation_list:
            pass
        else:
            formation_list.append(fm_name)

    return formation_list, rskey_updated


def __find_years(search_path, fm, rskey):
    years_list = []

    for file in glob.glob(search_path + fm[0] + "*max_" + rskey + "*.gri"):
        full_date = pathlib.Path(file).stem.split("--")[2]
        year = full_date[0:4]

        if year in years_list:
            pass
        else:
            years_list.append(year)

    return years_list


def __neigh_nodes(x):  # If all the four nodes of the cell are not masked we count the area
    sq_vert = {(x[0] + 1, x[1]), (x[0], int(x[1]) + 1), (x[0] + 1, x[1] + 1)}

    return sq_vert


def calc_plume_area(path, rskey):
    print("*** Calculating plume area for: " + rskey + " ***")

    formations, rskey_updated = np.array(__find_formations(path, rskey))
    print("Formations found: ", formations)

    years = np.array(__find_years(path, formations, rskey_updated))
    print("Dates found: ", years)

    # area_array = (product(formation, var, year))  # Not used anymore?

    var = "max_" + rskey_updated
    dict_out = []
    for fm in formations:
        for year in years:
            print(path + fm + "--" + var + "--" + year + "*.gri")
            path_file = glob.glob(path + fm + "--" + var + "--" + year + "*.gri")
            # path_file = path + fm + "--" + var + "--" + year + "0101.gri"
            mysurf = xtgeo.surface_from_file(path_file[0])
            use_nodes = np.ma.nonzero(mysurf.values)  # Indexes of the existing nodes
            use_nodes = set(list(tuple(zip(use_nodes[0], use_nodes[1]))))
            all_neigh_nodes = list(map(__neigh_nodes, use_nodes))
            test0 = [xx.issubset(use_nodes) for xx in all_neigh_nodes]
            dict_out_temp = [float(year), float(sum(t * mysurf.xinc * mysurf.yinc for t in test0)), fm]
            dict_out.append(dict_out_temp)

    return dict_out


def __read_args():
    args = __make_parser().parse_args()
    path = args.input

    if not os.path.isdir(path):
        text = f"Directory not found: {path}"
        raise FileNotFoundError(text)

    return path


def __convert_to_data_frame(results, rskey):
    # Convert into Pandas DataFrame
    df = pd.DataFrame.from_records(results, columns=["DATE", "AREA_"+rskey, "FORMATION_"+rskey])
    df = df.pivot(index="DATE", columns="FORMATION_"+rskey, values="AREA_"+rskey)
    df.reset_index(inplace=True)
    df.columns.name = None
    df.columns = [x + "_"+rskey if x != "DATE" else x for x in df.columns]
    return df


def main():
    path = __read_args()

    sgas_results = calc_plume_area(path, "sgas")
    if sgas_results:
        print("SGAS plume areas sucessfully collected.")

    amfg_results = calc_plume_area(path, "amfg")
    if amfg_results:
        print("AMFG plume areas sucessfully collected.")

    sgas_df = __convert_to_data_frame(sgas_results, "SGAS")
    amfg_df = __convert_to_data_frame(amfg_results, "AMFG")
    # Merge them together
    df = pd.merge(sgas_df, amfg_df)

    # Export to CSV
    # out = "share/results/tables/plumearea.csv"
    out = "plume_area.csv"
    df.to_csv(out, index=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())
