import json
import math
import operator
from operator import getitem

import numpy as np
import pandas as pd
import math
import progressbar

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


def prepare_matrix(filename):
    """
    Opens a json file and converts it to a dataframe of the correct format
    """
    df = pd.read_json(filename)
    df = df.dropna(axis=1, how="all")
    df_proc = df.T
    df_proc = df_proc.dropna()
    df_proc = df_proc.round(decimals=1)
    return df_proc


def normalise_matrix(df):
    """
    Performs normalisation calculations on each column of the matrix and returns
    the normalised matrix
    Pandas automatically applies colomn-wise functions.
    """
    normalized_df = df - df.min()
    normalized_df = normalized_df / normalized_df.sum()
    normalized_df = normalized_df.round(decimals=2)
    return normalized_df


def sd_matrix(df):
    """
    Calculates the Standard Deviation of each column and returns a dictionary 
    containing the positions with the most significant SD
    """
    df_std = df.std(axis=0, skipna=True)
    df_std = df_std[df_std > df_std.mean()]
    df_index = [ind for ind in df_std.index]
    return df_index


def gini_weight(df):
    d = {}
    for col in df.columns:
        col_list = df[col].to_numpy()
        mad = np.abs(np.subtract.outer(col_list, col_list)).mean()
        rmad = mad / col_list.mean()
        g1 = 0.5 * rmad
        d[col] = g1
        d_df = pd.DataFrame.from_dict(d, orient="index")
    return d_df


def weight_matrix(df):
    """
    Calculates the gini column weights and creates a new weighted PSSM
    """
    df_weighted = df * gini_weight(df)
    df_weighted = df_weighted.round(decimals=2)
    return df_weighted


def matrix_equal(df1, df2):
    """
    Returns a boolean whether two matrices are equal
    """
    return df2.equals(df1)


def sum_squared_distance_matrix(df1, df2):
    """
    Calculates the squared distances of two matrices and returns the sum value
    """
    adf1, adf2 = df1.align(df2, join="outer", axis=1)
    full_ssd = (adf1 - adf2) ** 2
    full_ssd = full_ssd.dropna(axis=1, how="all")
    full_ssd_val = full_ssd.fillna(0).values.sum()
    return full_ssd_val


def euclidian_distance(df1, df2):
    """
    Calculates the euclidian distance of the two matrices. Sort will be ascending.
    """
    ed_df = (df1 - df2) ** 2
    ed_df = ed_df.dropna(axis=1, how="all")
    full_eu = math.sqrt(ed_df.fillna(0).values.sum())
    return full_eu


def Kullback_Leibler_distance(df1, df2):
    """
    Calculates the Kullback-Leibler distance of the two matrices. 
    As defined in Aerts et al. (2003). Also called Mutual Information.
    Sort will be ascending.
    """
    kld_df = df1 * math.log(sum(df1 / df2))
    kld_df = kld_df.dropna(axis=1, how="all")
    full_kld = sum(kld_df)
    return full_kld


def correlation_coefficient(df1, df2):
    """
    Calculates and return the correlation coefficient of two matrices.
    Sort will be decreasing.
    """
    mean1 = sum(df1.mean())
    mean2 = sum(df2.mean())
    summerA = (df1 - mean1) * (df2 - mean2)
    summerB = (df1 - mean1) ** 2
    summerC = (df2 - mean2) ** 2
    return sum(summerA) / math.sqrt((sum(summerB) * sum(summerC)))


def calc_pearson_correlation(df1, df2):
    pearsons = []

    for i in range(0, len(df1.columns)):
        dfi = df1.iloc[:, i]
        dfj = df2.iloc[:, i]
        pearson = dfi.corr(dfj)
        pearson = pearson.round(decimals=3)

        # Turning the correlation coefficient scale from -1 - 1 to 0-1
        pearson_scale = (pearson + 1) / 2
        pearson_scale = pearson_scale.round(decimals=3)

        pearsons.append(pearson_scale)
    return pearsons


def calc_spearmans_correlation(df1,df2):
    spearmans = []

    for i in range(0, len(df1.columns)):
        dfi = df1.iloc[:, i]
        dfj = df2.iloc[:, i]
        spearman = dfi.corr(dfj, method='spearman')
        spearman = spearman.round(decimals=3)

        # Turning the correlation coefficient scale from -1 - 1 to 0-1
        spearmans_scale = (spearman + 1) / 2
        spearmans_scale = spearmans_scale.round(decimals=3)
        spearmans.append(spearmans_scale)
    return spearmans


def get_df_window(df1, df2, pep_window, i, j):
    a = df1.loc[:, i : i + pep_window]
    b = df2.loc[:, j : j + pep_window]

    a.columns = [ind for ind in range(0, len(a.columns))]
    b.columns = [ind for ind in range(0, len(b.columns))]

    return a, b


def compare_matrix_windowed(df1, df2, pep_window):
    """
    Compares two matrices using a window of comparison and returns a dictionary
    containing the positions of each matrix and the SSD
    """
    sdfs = {}
    ssd_sum_dic = {}
    pearsons = {}
    spearmans = {}

    it_window_a = len(df1.columns) - pep_window
    it_window_b = len(df2.columns) - pep_window

    for i in range(0, it_window_a):
        for j in range(0, it_window_b):
            # print("===> i: {}:{} j: {}:{}".format(i, i + pep_window, j, j + pep_window))
            a, b = get_df_window(df1, df2, pep_window, i, j)

            a_gini = gini_weight(a).apply(lambda x: 1 - x)
            a_gini_df = a_gini.T
            b_gini = gini_weight(b).apply(lambda x: 1 - x)
            b_gini_df = b_gini.T

            a_gini_df = pd.DataFrame(np.repeat(a_gini_df.values, len(a.index), axis=0))
            b_gini_df = pd.DataFrame(np.repeat(b_gini_df.values, len(b.index), axis=0))

            a_all = a * a_gini_df.values
            b_all = a * b_gini_df.values

            ssd = (a - b) ** 2
            ssd_sum = ssd.sum()
            ssd_sum = pd.DataFrame(ssd_sum)
            ssd_sum = ssd_sum.T
            ssd_sum = ssd_sum.rename(index={0: "SSD"})

            # 1st way
            sdf = (a_all - b_all) ** 2
            sdf = sdf[sdf < sdf.mean()].fillna(1)

            pearsons_cor = calc_pearson_correlation(a, b)
            spearmans_cor = calc_spearmans_correlation(a, b)

            # Suggested way:
            # SDF as sum(SSD * (1 - gini1) * (1 - gini2))
            # and sum(SSD * (1 - gini1 * gini2))

            # Suggested way 1:
            # sdf = (ssd_sum * a_gini_df.values * b_gini_df.values)

            # Suggested way 2:
            # sdf = (ssd_sum * (1-(gini_weight(a).T.values * gini_weight(b).T.values)**2))
            # print("{}:{} - {}:{}".format(i, i + pep_window, j, j + pep_window))
            # print("this is ssd: ",ssd_sum)
            # print("this is gini a ",gini_weight(a).T)
            # print("this is b " , gini_weight(b).T)

            # print("this is sdf: ",sdf)
            # print("final sdf:",sdf.fillna(1).values.sum().round(decimals=3))
            key = "{}:{} - {}:{}".format(i, i + pep_window, j, j + pep_window)
            sdfs[key] = (
                sdf.fillna(1).values.sum().round(decimals=3)
            )
            ssd_sum_dic[key] = ssd_sum
            pearsons[key] = pearsons_cor
            spearmans[key] = spearmans_cor

        sdfs_sorted = sorted(sdfs.items(), key=operator.itemgetter(1))

    sdf_0 = sdfs_sorted[0]
    if sdf_0[0] in ssd_sum_dic:
        ssd_print = ssd_sum_dic[sdf_0[0]]
        pearsons_print = pearsons[sdf_0[0]]
        spearmans_print = spearmans[sdf_0[0]]
    #   print(ssd_print)

    return sdfs_sorted, ssd_print, pearsons_print, spearmans_print


def compare_two_files(base_file, second_file, pep_window):
    """
    Calculate all the comparisons for two PSSMs
    """
    df1 = prepare_matrix(base_file)
    df1_norm = normalise_matrix(df1)
    df1_sd = sd_matrix(df1)
    # df1_weigthed = weight_matrix(df1_norm)

    df2 = prepare_matrix(second_file)
    df2_sd = sd_matrix(df2)
    df2_norm = normalise_matrix(df2)
    # df2_weigthed = weight_matrix(df2_norm)

    ssd = sum_squared_distance_matrix(df1_norm, df2_norm)
    equality = matrix_equal(df1_norm, df2_norm)
    sdfs, ssd_print, pearsons, spearmans = compare_matrix_windowed(df1_norm, df2_norm, pep_window)

    return equality, df1_sd, df2_sd, ssd, sdfs, df1_norm, df2_norm, ssd_print, pearsons, spearmans


def compare_combined_file(base_file, combined_file, pep_window):
    """
    Calculate all the comparisons for a PSSM and a .json file cointaing multiple
    PSSMs
    """

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
                (
                    equality,
                    _,
                    _,
                    ssd,
                    sdfs,
                    df1,
                    df2,
                    ssd_print,
                    pearsons,
                    spearmans,
                ) = compare_two_files(base_file, json_pssm, pep_window)
                res["equality"] = equality
                res["base"] = base_file
                res["second"] = data[pssm]["motif"]
                res["ssd"] = ssd
                res["sdf"] = sdfs[0]
                res["df1"] = df1
                res["df2"] = df2
                res["ssd_print"] = ssd_print
                res["pearsons"] = pearsons
                res["spearmans"] = spearmans
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


def print_df_ranges(df, region, ssd_print, pearsons, spearmans):
    a, b = region.split(":")
    gini = gini_weight(df)
    gini = gini.T
    new_df = df.append(gini)
    new_df = new_df.rename(index={0: "Gini"})

    column_list = list(range(int(a), int(b) + 1))
    ssd_print.columns = column_list
    new_df = new_df.append(ssd_print)

    pearsons_df = pd.DataFrame(pearsons)
    pearsons_df = pearsons_df.T
    pearsons_df = pearsons_df.rename(index={0: "Pearson"})
    pearsons_df.columns = column_list
    new_df = new_df.append(pearsons_df)

    spearmans_df = pd.DataFrame(spearmans)
    spearmans_df = spearmans_df.T
 
    spearmans_df = spearmans_df.rename(index={0: "Spearman"})
    spearmans_df.columns = column_list
    new_df = new_df.append(spearmans_df)

    print(new_df.loc[:, int(a) : int(b)])
