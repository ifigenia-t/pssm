import json
import math
import operator
from operator import getitem

import numpy as np
import pandas as pd
import progressbar


def prepare_matrix(filename):
    '''
    Opens a json file and converts it to a dataframe of the correct format
    '''
    df = pd.read_json(filename)
    df = df.dropna(axis=1, how="all")
    df_proc = df.T
    df_proc = df_proc.dropna()
    df_proc = df_proc.round(decimals=1)
    return df_proc


def sd_matrix(df):
    '''
    Calculates the Standard Deviation of each column and returns a dictionary 
    containing the positions with the most significant SD
    '''
    df_std = df.std(axis=0, skipna=True)
    df_std = df_std[df_std > df_std.mean()]
    df_index = [ind for ind in df_std.index]
    return df_index


def weight_matrix(df):
    '''
    Calculates the gini column weights and creates a new weighted PSSM
    '''
    mad = abs(np.subtract.outer(df, df)).mean()
    rmad = mad / df.mean()
    g1 = 0.5 * rmad
    df_weighted = df * g1
    return df_weighted


def matrix_equal(df1, df2):
    '''
    Returns a boolean whether two matrices are equal
    '''
    return df2.equals(df1)


def sum_squared_distance_matrix(df1, df2):
    '''
    Calculates the squared distances of two matrices and returns the sum value
    '''
    adf1, adf2 = df1.align(df2, join="outer", axis=1)
    full_ssd = (adf1 - adf2) ** 2
    full_ssd = full_ssd.dropna(axis=1, how="all")
    full_ssd_val = full_ssd.fillna(0).values.sum()
    return full_ssd_val


def compare_matrix_windowed(df1, df2, pep_window):
    '''
    Compares two matrices using a window of comparison and returns a dictionary
    containing the positions of each matrix and the SSD
    '''
    sdfs = {}

    it_window_a = len(df1.columns) - pep_window
    it_window_b = len(df2.columns) - pep_window

    for i in range(0, it_window_a):
        for j in range(0, it_window_b):

            # print("===> i: {}:{} j: {}:{}".format(i, i + pep_window, j, j + pep_window))

            a = df1.loc[:, i : i + pep_window]
            b = df2.loc[:, j : j + pep_window]

            a.columns = [ind for ind in range(0, len(a.columns))]
            b.columns = [ind for ind in range(0, len(b.columns))]

            sdf = (a - b) ** 2

            sdfs["{}:{} - {}:{}".format(i, i + pep_window, j, j + pep_window)] = (
                sdf.fillna(0).values.sum().round(decimals=3)
            )

        sdfs_sorted = sorted(sdfs.items(), key=operator.itemgetter(1))

    return sdfs_sorted


def compare_two_files(base_file, second_file, pep_window):
    '''
    Calculate all the comparisons for two PSSMs
    '''
    df1 = prepare_matrix(base_file)
    df1_sd = sd_matrix(df1)
    df1_weigthed = weight_matrix(df1)

    df2 = prepare_matrix(second_file)
    df2_sd = sd_matrix(df2)
    df2_weigthed = weight_matrix(df2)

    ssd = sum_squared_distance_matrix(df1_weigthed, df2_weigthed)
    equality = matrix_equal(df1_weigthed, df2_weigthed)
    sdfs = compare_matrix_windowed(df1_weigthed, df2_weigthed, pep_window)

    return equality, df1_sd, df2_sd, ssd, sdfs


def compare_combined_file(base_file, combined_file, pep_window):
    '''
    Calculate all the comparisons for a PSSM and a .json file cointaing multiple
    PSSMs
    '''

    with open(combined_file) as json_file:
        data = json.load(json_file)
        bar = progressbar.ProgressBar(
            maxval=len(data),
            widgets=[progressbar.Bar("=", "[", "]"), " ", progressbar.Percentage()],
        )

        results = []
        bar.start()
        i = 0

        for pssm in data:
            res = {}
            try:
                json_pssm = json.dumps(data[pssm]["pssm"])
                equality, _, _, ssd, sdfs = compare_two_files(
                    base_file, json_pssm, pep_window
                )
                res["equality"] = equality
                res["base"] = base_file
                res["second"] = data[pssm]["motif"]
                res["ssd"] = ssd
                res["sdf"] = sdfs[0]
                results.append(res)

            except TypeError as ex:
                print("error: {} on pssm: {}".format(ex, pssm))

            except IndexError as ex:
                print("error: {} on pssm: {}, res: {} ".format(ex, pssm, res))

            i += 1
            bar.update(i)

        bar.finish()

        results.sort(key=lambda x: x["sdf"][1])
        
        return results
