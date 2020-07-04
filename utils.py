import math
import operator

import numpy as np
import pandas as pd


def prepare_matrix(filename):
    df = pd.read_json(filename)
    df = df.dropna(axis=1, how="all")
    df_proc = df.T
    df_proc = df_proc.dropna()
    df_proc = df_proc.round(decimals=1)
    return df_proc


def sd_matrix(df):
    df_std = df.std(axis=0, skipna=True)
    df_std = df_std[df_std > df_std.mean()]
    df_index = [ind for ind in df_std.index]
    return df_index


def weight_matrix(df):
    mad = abs(np.subtract.outer(df, df)).mean()
    rmad = mad / df.mean()
    g1 = 0.5 * rmad
    df_weighted = df * g1
    return df_weighted


def matrix_equal(df1, df2):
    return df2.equals(df1)


def sum_squared_distance_matrix(df1, df2):
    adf1, adf2 = df1.align(df2, join="outer", axis=1)
    full_sdf = (adf1 - adf2) ** 2
    full_sdf = full_sdf.dropna(axis=1, how="all")
    full_sdf_val = full_sdf.fillna(0).values.sum()
    return full_sdf_val


def compare_matrix_windowed(df1, df2, pep_window):
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


def compare_files(file1, file2, pep_window):
    df1 = prepare_matrix(file1)
    df1_sd = sd_matrix(df1)
    df1_weigthed = weight_matrix(df1)

    df2 = prepare_matrix(file2)
    df2_sd = sd_matrix(df2)
    df2_weigthed = weight_matrix(df2)

    sdf = sum_squared_distance_matrix(df1_weigthed, df2_weigthed)
    equality = matrix_equal(df1_weigthed, df2_weigthed)
    sdfs = compare_matrix_windowed(df1_weigthed, df2_weigthed, pep_window)

    return equality, df1_sd, df2_sd, sdf, sdfs

