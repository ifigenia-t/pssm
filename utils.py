import json
import math
import operator
from operator import getitem

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from tqdm.auto import tqdm

from data_prep import *

pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)

buffer = 1
gini_cutoff = 0.57


class NoResultsError(Exception):
    pass


# Importance Metric
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
    """
    Calculates the Gini Coefficient which is a measure of statistical dispersion.
    Gini = 1 means maximal inequallity
    Gini = 0 means perfect equallity where all the values are the same. 
    """
    d = {}
    # df = df ** 2
    for col in df.columns:
        col_list = df[col].to_numpy()
        mad = np.abs(np.subtract.outer(col_list, col_list)).mean()
        rmad = mad / col_list.mean()
        g1 = 0.5 * rmad
        d[col] = g1
        d_df = pd.DataFrame.from_dict(d, orient="index")
        d_df = d_df.round(decimals=2)
    return d_df


def weight_matrix(df):
    """
    Calculates the gini column weights and creates a new weighted PSSM
    """
    df_weighted = df * gini_weight(df)
    df_weighted = df_weighted.round(decimals=2)
    return df_weighted


# Calculation of the peptide window
def calc_brute_force_window(df1, df2):
    """
    Calculates a list of possible windows for the comparison of two PSSMs.
    """
    max_diff = 0
    if len(df1.columns) > len(df2.columns):
        max_diff = len(df2.columns)
    else:
        max_diff = len(df1.columns)
    return [x for x in range(1, max_diff + 1)]

def gini_window_index(df):
    """
    Finds all the important positions of a PSSM. Important positions are all
    the positions that have a gini larger than the (mean + SD) of all the ginis.
    """
    gini_df = gini_weight(df)
    # gini_df_sorted = gini_df.sort_values(by=0, ascending=False)
    # gini_window = []
    # select_indices = [0, 1]
    # gini_window = gini_df_sorted.iloc[select_indices].index.tolist()
    
    # gini_window_index = [ind for ind in gini_window.index]
    # print("This is the gini window index ", gini_window_index)
    gini_window = gini_df[gini_df > gini_cutoff]
    gini_window.dropna(inplace=True)
    # print("This is the gini_window: ",gini_window)
    if len(gini_window) == 0:
        gini_df_sorted = gini_df.sort_values(by=0, ascending=False)
        gini_window = []
        select_indices = [0, 1]
        gini_window_index = gini_df_sorted.iloc[select_indices].index.tolist()
        # print("This is the new gini index: ", gini_window_index, " calc with 2 max")
    else:
        gini_window_index = [ind for ind in gini_window.index]
        # print("This is the gini index: ", gini_window_index, " calc with gini cutoff")
    
        # gini_window_index.sort()
    
    # gini_window_index = [ind for ind in gini_window.index]
    # print("This is the gini window index ", gini_window_index)
    # # if len(gini_window) == 0:
    # #     df = df ** 2
    # #     gini_df = gini_weight(df)
    # #     gini_window = gini_df[gini_df > gini_df.mean() + gini_df.std()]
    # #     gini_window.dropna(inplace=True)
    # #     if len(gini_window) == 0:
    # #         gini_window = gini_df[gini_df > gini_df.mean()]
    # #         gini_window.dropna(inplace=True)
    # gini_window_index = [ind for ind in gini_window.index]
    # if len(gini_window_index) == 0:
    #     gini_window = gini_df[gini_df > gini_df.mean()]
    #     gini_window.dropna(inplace=True)
    #     gini_window_index = [ind for ind in gini_window.index]
    return gini_window_index


def calc_gini_windows(df1, df2):
    """
    Calculates the list of all the windows for the comparison of the 2 PSSMs,
    according to their important positions.
    """
    index_1 = gini_window_index(df1)
    index_2 = gini_window_index(df2)
    # sdt_deviation = sd_matrix(df1)
    print("Index_1: ", index_1)
    print("Index_2: ", index_2)

    windows_list = []
    # if len(df1.columns) <= len(df2.columns):
    #     window = len(df1.columns) 
    # else:
    #     window = len(df2.columns)
    # windows_list.append(window)
    windows_list = calc_brute_force_window(df1, df2)
    # if len(index_1) != 0 and len(index_2) != 0:
    #     if len(index_1) == 1:
    #         window = max(index_2) - min(index_2) + buffer
    #         windows_list.append(window)
    #     elif len(index_2) == 1:
    #         window = max(index_1) - min(index_1) + buffer
    #         windows_list.append(window)
    #     else:
    #         if len(df1.columns) <= len(df2.columns):
    #             min_window = max(index_1) - min(index_1) + buffer
    #             # min_window = max(sdt_deviation) - min(sdt_deviation) + buffer
    #             max_window = max(index_2) - min(index_2) + buffer
    #             if min_window > max_window:
    #                 max_window, min_window = min_window, max_window
    #         else:
    #             min_window = max(index_2) - min(index_2) + buffer
    #             max_window = max(index_1) - min(index_1) + buffer
    #             if min_window > max_window:
    #                 max_window, min_window = min_window, max_window
    #         windows_list = [x for x in range(min_window, max_window + 1)]
    # elif len(index_1) == 0 or len(index_2) == 0:
    #     cindex = index_1 + index_2
    #     max_window = min_window = max(cindex) - min(cindex) + buffer
    #     windows_list = [x for x in range(min_window, max_window + 1)]
    # else:
    #     windows_list = calc_brute_force_window(df1, df2)
   
    print("This is the windows_list: ", windows_list)
    return windows_list


def get_df_window(df1, df2, pep_window, i, j):
    """
    Slices the dataframes according to a given window size.
    """
    a = df1.loc[:, i : i + pep_window - 1]
    b = df2.loc[:, j : j + pep_window - 1]

    a.columns = [ind for ind in range(0, len(a.columns))]
    b.columns = [ind for ind in range(0, len(b.columns))]

    return a, b


def find_motif(df):
    """
    Finds the motif of the pssm using the important positions.
    """
    motif = ""
    motif_l = []
    gini_index = gini_window_index(df)
    st_index = sd_matrix(df)
    if len(gini_index) != 0:
        motif_range = [x for x in range(min(gini_index), max(gini_index) + 1)]
    else:
        motif_range = [x for x in range(0, len(df) + 1)]
    for col in motif_range:
        if col in st_index:
            Index_label = df[df[col] > df[col].mean()].index.tolist()
            if len(Index_label) == 1:
                motif_l.append(Index_label[0])
            else:
                Index_label = df[df[col] == df[col].max()].index.tolist()
                motif_l.append(Index_label[0])
        else:
            motif_l.append("x")
    print("This is the motif: ", motif.join(motif_l))
    return motif.join(motif_l)


# Similarity Metrics
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


def calc_sum_of_squared_distance(dfi, dfj):
    """
    Calculates the square distance between 2 dataframes and returns their sum.
    Order is ascending.
    """
    ssd = (dfi - dfj) ** 2
    ssd_sum = ssd.sum().round(decimals=3)

    return ssd_sum


def calc_dot_product(dfi, dfj):
    """
    Calculates the dot product between 2 dataframes and returns their sum.
    Order is ascending.
    """
    dot_product = dfi.values * dfj.values
    dot_product_sum = dot_product.sum().round(decimals=3)

    return dot_product_sum


def calc_Kullback_Leibler_distance(dfi, dfj):
    """
    Calculates the Kullback-Leibler distance of the two matrices. 
    As defined in Aerts et al. (2003). Also called Mutual Information.
    Sort will be ascending.
    Epsilon is used here to avoid conditional code for checking that neither P nor Q is equal to 0.
    """
    epsilon = 0.00001

    P = dfi + epsilon
    Q = dfj + epsilon
    divergence = np.sum(P * np.log2(P / Q))

    return divergence


def calc_pearson_correlation(dfi, dfj):
    """
    Calculates Pearson's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1. 
    Order is decreasing.
    """
    pearson = dfi.corr(dfj)
    pearson = pearson.round(decimals=3)

    # Turning the correlation coefficient scale from -1 - 1 to 0-1
    # pearson_scale = (pearson + 1) / 2
    # pearson_scale = pearson_scale.round(decimals=3)

    return pearson


def calc_spearmans_correlation(dfi, dfj):
    """
    Calculates the Spearman's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1. 
    Order is decreasing.
    """
    spearman = dfi.corr(dfj, method="spearman")
    spearman = round(spearman, 3)

    # Turning the correlation coefficient scale from -1 - 1 to 0-1
    # spearmans_scale = (spearman + 1) / 2
    # spearmans_scale = round(spearmans_scale, 3)

    return spearman


def calc_kendall_correlation(dfi, dfj):
    """
    Calculates Kendall's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1. 
    Order is decreasing.
    """
    kendall = dfi.corr(dfj, method="kendall")
    kendall = round(kendall, 3)

    # Turning the correlation coefficient scale from -1 - 1 to 0-1
    # kendalls_scale = (kendall + 1) / 2
    # kendalls_scale = round(kendalls_scale, 3)

    return kendall


def calculate_similarity(df1, df2):
    """
    Calculates all the similarity measures column wise and returns 1 row dataframes.
    """
    kendalls = []
    pearsons = []
    spearmans = []
    dots = []
    ssds = []
    kls = []

    for i in range(0, len(df1.columns)):
        dfi = df1.iloc[:, i]
        dfj = df2.iloc[:, i]

        kendall = calc_kendall_correlation(dfi, dfj)
        pearson = calc_pearson_correlation(dfi, dfj)
        spearman = calc_spearmans_correlation(dfi, dfj)
        dot_product = calc_dot_product(dfi, dfj)
        ssd = calc_sum_of_squared_distance(dfi, dfj)
        kl = calc_Kullback_Leibler_distance(dfi, dfj)

        kendalls.append(kendall)
        pearsons.append(pearson)
        spearmans.append(spearman)
        dots.append(dot_product)
        ssds.append(ssd)
        kls.append(kl)

    kendalls_df = convert_to_df(kendalls, "Kendall")
    pearsons_df = convert_to_df(pearsons, "Pearson")
    spearmans_df = convert_to_df(spearmans, "Spearman")
    dots_df = convert_to_df(dots, "Dot")
    ssds_df = convert_to_df(ssds, "SSD")
    kls_df = convert_to_df(kls, "KL")

    return kendalls_df, pearsons_df, spearmans_df, dots_df, ssds_df, kls_df


# Comparisons


def compare_matrix_windowed(df1, df2, pep_window):
    """
    Compares two matrices using a window of comparison and returns a dictionary
    containing the positions of each matrix and the SSD
    """
    results = {}
    results_sorted = None
    ssd_sum_dic = {}
    pearsons = {}
    spearmans = {}
    kendalls = {}
    dot_products = {}
    kl_divergence = {}
    ginis_1 = {}
    ginis_2 = {}

    it_window_a = len(df1.columns) - pep_window
    it_window_b = len(df2.columns) - pep_window

    for i in range(0, it_window_a + 1):
        for j in range(0, it_window_b + 1):
            key = "{}:{} - {}:{}".format(i, i + pep_window - 1, j, j + pep_window - 1)
            a, b = get_df_window(df1, df2, pep_window, i, j)

            a_gini_df = gini_weight(a)
            a_gini_df = a_gini_df.T
            b_gini_df = gini_weight(b)
            b_gini_df = b_gini_df.T

            (
                kendalls_cor,
                pearsons_cor,
                spearmans_cor,
                dot_product,
                ssd,
                kl,
            ) = calculate_similarity(a, b)
            # TODO make this configurable
            comparison = pd.DataFrame(
                pearsons_cor.values * a_gini_df.values * b_gini_df.values,
                columns=pearsons_cor.columns,
                index=pearsons_cor.index,
            )

            # Suggested way:
            # SDF as sum(SSD * (1 - gini1) * (1 - gini2))
            # and sum(SSD * (1 - gini1 * gini2))

            results[key] = comparison.values.sum().round(decimals=3)
            ssd_sum_dic[key] = ssd
            pearsons[key] = pearsons_cor
            spearmans[key] = spearmans_cor
            kendalls[key] = kendalls_cor
            dot_products[key] = dot_product
            kl_divergence[key] = kl
            ginis_1[key] = a_gini_df
            ginis_2[key] = b_gini_df

        # TODO make order cofigurable
        results_sorted = sorted(
            results.items(), key=operator.itemgetter(1), reverse=True
        )

    if results_sorted is None:
        raise NoResultsError(
            "window a: {} , window b: {}".format(it_window_a, it_window_b)
        )

    res_0 = results_sorted[0]

    if res_0[0] in ssd_sum_dic:
        ssd_res = ssd_sum_dic[res_0[0]]
        pearsons_res = pearsons[res_0[0]]
        spearmans_res = spearmans[res_0[0]]
        kendalls_res = kendalls[res_0[0]]
        dot_product_res = dot_products[res_0[0]]
        kl_divergence_res = kl_divergence[res_0[0]]
        ginis_1_res = ginis_1[res_0[0]]
        ginis_2_res = ginis_2[res_0[0]]

    return (
        results_sorted,
        ssd_res,
        pearsons_res,
        spearmans_res,
        kendalls_res,
        dot_product_res,
        kl_divergence_res,
        ginis_1_res,
        ginis_2_res,
    )


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

    ssd_global = sum_squared_distance_matrix(df1_norm, df2_norm)
    equality = matrix_equal(df1_norm, df2_norm)
    (
        comparison_results,
        ssd,
        pearsons,
        spearmans,
        kendalls,
        dot_products,
        kl_divergence,
        gini_a,
        gini_b,
    ) = compare_matrix_windowed(df1_norm, df2_norm, pep_window)

    return (
        equality,
        df1_sd,
        df2_sd,
        ssd_global,
        comparison_results,
        df1_norm,
        df2_norm,
        ssd,
        pearsons,
        spearmans,
        kendalls,
        dot_products,
        kl_divergence,
        gini_a,
        gini_b,
    )


def compare_combined_file(base_file, combined_file, pep_window):
    """
    Calculate all the comparisons for a PSSM and a .json file cointaing multiple
    PSSMs
    """
    results = []
    one_window = 0
    with open(combined_file) as json_file:
        data = json.load(json_file)

        # TODO find a way not have to use this
        index_names = []
        for elm in data:
            index_names.append(data[elm]["motif"])
        # "ELM",
        col_names = [
            "Quality",
            "No Important Positions",
            "Windows",
            "Comparison Results",
            "Norm. Window",
            "Motif 1",
            "Motif 2",
            "Consensus",
            "Comparison Windows",
            "Gini 1",
            "Gini 2",
            "Similarity",
        ]
        df = pd.DataFrame(columns=col_names, index=index_names)
        # df = pd.DataFrame(columns=col_names)
        # i = 0

        for pssm in tqdm(data):
            try:
                # json_pssm = json.dumps(data[pssm]["pssm"])
                json_pssm = json.dumps(data[pssm]["other_scoring_methods"]["log odds"])
                print("-----> ", data[pssm]["motif"])

                pep_windows = []
                if pep_window is 0:
                    df1 = prepare_matrix(base_file)
                    df1 = normalise_matrix(df1)
                    df2 = prepare_matrix(json_pssm)
                    df2 = normalise_matrix(df2)
                    gini_1 = gini_weight(df1)
                    gini_2 = gini_weight(df2)
                    pep_windows = calc_gini_windows(df1, df2)
                    index_1 = gini_window_index(df1)
                    # std_dev = sd_matrix(df1)
                    min_window = max(index_1) - min(index_1) + 1
                    # min_window_dev = max(std_dev) - min(std_dev) +1
                    # if min_window > min_window_dev:
                    #     min_window = min_window_dev
                    print("min_window is ", min_window)
                    if min_window > len(df2.columns):
                        min_window = len(df2.columns)
                        print("Or better the window is ", min_window)
                    index_2 = gini_window_index(df2)
                    if len(index_2) == 1:
                        one_window += 1
                    motif_1 = find_motif(df1)
                    motif_2 = find_motif(df2)
                else:
                    pep_windows.append(int(pep_window))

                print("pep_windows: ", pep_windows)

                for window in pep_windows:
                    # i += 1
                    if window >= min_window:
                        res = {}
                        (
                            equality,
                            _,
                            _,
                            ssd_global,
                            comparison_results,
                            df1,
                            df2,
                            ssd,
                            pearsons,
                            spearmans,
                            kendalls,
                            dot_products,
                            kl_divergence,
                            gini_a,
                            gini_b,
                        ) = compare_two_files(base_file, json_pssm, window)
                        res["equality"] = equality
                        res["base"] = base_file
                        res["second"] = data[pssm]["motif"]
                        res["ssd_global"] = ssd_global
                        res["comparison_results"] = comparison_results[0]
                        res["df1"] = df1
                        res["df2"] = df2
                        res["ssd"] = ssd
                        res["pearsons"] = pearsons
                        res["spearmans"] = spearmans
                        res["kendalls"] = kendalls
                        res["dot_products"] = dot_products
                        res["kl_divergence"] = kl_divergence
                        res["pep_window"] = window
                        res["norm_window"] = comparison_results[0][1] / window
                        res["gini_1"] = gini_a
                        res["gini_2"] = gini_b
                        res["motif_1"] = motif_1
                        res["motif_2"] = motif_2

                        results.append(res)
                        print(
                            "second: ",
                            res["second"],
                            "window ",
                            window,
                            " comparison ",
                            comparison_results[0][1],
                        )
                        print("Norm_window: ", res["norm_window"])


                        if (
                            res["norm_window"] > df.at[res["second"], "Norm. Window"]
                        ) or pd.isna(df.at[res["second"], "Norm. Window"]):
                            print("adding...")
                            df.loc[res["second"]] = [
                                data[pssm]["quality"],
                                len(index_2),
                                window,
                                comparison_results[0][1],
                                res["norm_window"],
                                res["motif_1"],
                                res["motif_2"],
                                data[pssm]["consensus"],
                                comparison_results[0][0],
                                convert_df_to_string(res["gini_1"]),
                                convert_df_to_string(res["gini_2"]),
                                convert_df_to_string(res["pearsons"])
                            ]

                        # df.loc[i] = [
                        #     res["second"],
                        #     data[pssm]["quality"],
                        #     len(index_2),
                        #     window,
                        #     comparison_results[0][1],
                        #     res["norm_window"],
                        #     res["motif_1"],
                        #     res["motif_2"],
                        #     data[pssm]["consensus"],
                        #     comparison_results[0][0],
                        #     convert_df_to_string(res["gini_1"]),
                        #     convert_df_to_string(res["gini_2"]),
                        #     convert_df_to_string(res["pearsons"]),
                        # ]

            except TypeError as ex:
                print("error: {} on pssm: {}".format(ex, pssm))
            except IndexError as ex:
                print("error: {} on pssm: {}, res: {} ".format(ex, pssm, res))
            except NoResultsError as ex:
                print("error: {} on pssm: {}".format(ex, pssm))
                continue

        df = df.sort_values(by=["Norm. Window"], ascending=False)
        df.to_csv("TANK_vs_ELM_min_len_-.csv")

        results.sort(key=lambda x: float(x["norm_window"]), reverse=True)
        print("Results with 1 important position: ", one_window)
        for res in results:
            print(
                "second: ",
                res["second"],
                "windows: ",
                comparison_results[0][0],
                "Norm Window Values: ",
                res["norm_window"],
                "Original window: ",
                res["pep_window"],
            )
        print("\n")

        return results


def compare_two_combined(file_1, file_2, pep_window):

    results = []
    one_window = 0
    with open(file_1) as json_file_1, open(file_2) as json_file_2:

        data1 = json.load(json_file_1)
        data2 = json.load(json_file_2)

        # TODO find a way not have to use this
        index_names = []
        for pssm in data1:
            index_names.append(data1[pssm]["motif"])

        col_names = [
            "ELM",
            "Quality",
            "No Important Positions",
            "Windows",
            "Comparison Results",
            "Norm. Window",
            "Motif 1",
            "Motif 2",
            "Consensus",
            "Comparison Windows",
            "Gini 1",
            "Gini 2",
            "Similarity",
        ]
        df = pd.DataFrame(columns=col_names, index=index_names)
        # df = pd.DataFrame(columns=col_names)
        # i = 0

        for base_pssm in tqdm(data1):
            base_file = json.dumps(data1[base_pssm]["pssm"])
            print("-----> ", data1[base_pssm]["motif"])
            for pssm in tqdm(data2):
                try:
                    # json_pssm = json.dumps(data2[pssm]["pssm"])
                    json_pssm = json.dumps(data2[pssm]["other_scoring_methods"]["log odds"])
                    
                    print("-----> ", data2[pssm]["motif"])

                    pep_windows = []
                    if pep_window is 0:
                        df1 = prepare_matrix(base_file)
                        df1 = normalise_matrix(df1)
                        df2 = prepare_matrix(json_pssm)
                        df2 = normalise_matrix(df2)
                        gini_1 = gini_weight(df1)
                        gini_2 = gini_weight(df2)
                        pep_windows = calc_gini_windows(df1, df2)
                        print("Windows ", pep_windows)
                        index_1 = gini_window_index(df1)
                        # std_dev = sd_matrix(df1)
                        min_window = max(index_1) - min(index_1) + 1
                        # min_window_dev = max(std_dev) - min(std_dev) +1
                        # if min_window > min_window_dev:
                        #     min_window = min_window_dev
                        print("min_window is ", min_window)
                        if min_window > len(df2.columns):
                            min_window = len(df2.columns)
                            print("Or better the window is ", min_window)
                        index_2 = gini_window_index(df2)
                        if len(index_2) == 1:
                            one_window += 1
                        motif_1 = find_motif(df1)
                        motif_2 = find_motif(df2)
                    else:
                        pep_windows.append(int(pep_window))

                    print("pep_windows: ", pep_windows)

                    for window in pep_windows:
                        # i += 1
                        if window >= min_window:
                            res = {}
                            (
                                equality,
                                _,
                                _,
                                ssd_global,
                                comparison_results,
                                df1,
                                df2,
                                ssd,
                                pearsons,
                                spearmans,
                                kendalls,
                                dot_products,
                                kl_divergence,
                                gini_a,
                                gini_b,
                            ) = compare_two_files(base_file, json_pssm, window)
                            res["equality"] = equality
                            res["base"] = base_file
                            res["base_name"] = data1[base_pssm]["motif"]
                            res["second"] = data2[pssm]["motif"]
                            res["ssd_global"] = ssd_global
                            res["comparison_results"] = comparison_results[0]
                            res["df1"] = df1
                            res["df2"] = df2
                            res["ssd"] = ssd
                            res["pearsons"] = pearsons
                            res["spearmans"] = spearmans
                            res["kendalls"] = kendalls
                            res["dot_products"] = dot_products
                            res["kl_divergence"] = kl_divergence
                            res["pep_window"] = window
                            res["norm_window"] = comparison_results[0][1] / window
                            res["gini_1"] = gini_a
                            res["gini_2"] = gini_b
                            res["motif_1"] = motif_1
                            res["motif_2"] = motif_2

                            results.append(res)
                            print(
                                "second: ",
                                res["second"],
                                "window ",
                                window,
                                " comparison ",
                                comparison_results[0][1],
                            )
                            print("Norm_window: ", res["norm_window"])
                            print(".........",res["base_name"])
                            print("....", pd.isna(df.at[res["base_name"], "Norm. Window"]))


                            if (
                                res["norm_window"] > df.at[res["base_name"], "Norm. Window"]
                            ) or pd.isna(df.at[res["base_name"], "Norm. Window"]):
                                print("adding...")
                                df.loc[res["base_name"]] = [
                                    res["second"],
                                    data2[pssm]["quality"],
                                    len(index_2),
                                    window,
                                    comparison_results[0][1],
                                    res["norm_window"],
                                    res["motif_1"],
                                    res["motif_2"],
                                    data2[pssm]["consensus"],
                                    comparison_results[0][0],
                                    convert_df_to_string(res["gini_1"]),
                                    convert_df_to_string(res["gini_2"]),
                                    convert_df_to_string(res["pearsons"]),
                                ]
                            # df.loc[i] = [
                            #     base_pssm,
                            #     res["second"],
                            #     data2[pssm]["quality"],
                            #     len(index_2),
                            #     window,
                            #     comparison_results[0][1],
                            #     res["norm_window"],
                            #     res["motif_1"],
                            #     res["motif_2"],
                            #     data2[pssm]["consensus"],
                            #     comparison_results[0][0],
                            #     convert_df_to_string(res["gini_1"]),
                            #     convert_df_to_string(res["gini_2"]),
                            #     convert_df_to_string(res["pearsons"]),
                            # ]

                except TypeError as ex:
                    print("error: {} on pssm: {}".format(ex, pssm))
                except IndexError as ex:
                    print("error: {} on pssm: {}, res: {} ".format(ex, pssm, res))
                except NoResultsError as ex:
                    print("error: {} on pssm: {}".format(ex, pssm))
                    continue
          
        df = df.sort_values(by=["Norm. Window"], ascending=False)
        df.to_csv("ProPD_vs_ELM_logodds_-1-1.csv")

        results.sort(key=lambda x: float(x["norm_window"]), reverse=True)
        print("Results with 1 important position: ", one_window)
        for res in results:
            print(
                "ProPD: ",
                res["base_name"],
                "ELM: ",
                res["second"],
                "windows: ",
                comparison_results[0][0],
                "Norm Window Values: ",
                res["norm_window"],
                "Original window: ",
                res["pep_window"],
            )
        print("\n")

        return results


def compare_single_file(single_file, pep_window):
    """
    Takes a single file with multiple PSSMs and compares them all with each other.
    It returns a dictionary with all the scores.
    """
    with open(single_file) as json_file:
        data = json.load(json_file)
        results = []

        col_names = []
        for col in data:
            col_names.append(data[col]["motif"])
        df = pd.DataFrame(columns=col_names, index=col_names)

        for pssm_i in tqdm(data):
            json_pssm_1 = json.dumps(data[pssm_i]["pssm"])
            row = {}
            for pssm_j in tqdm(data):
                json_pssm_2 = json.dumps(data[pssm_j]["pssm"])
                print(data[pssm_i]["motif"], " -----> ", data[pssm_j]["motif"])

                pep_windows = []
                if pep_window is 0:
                    df1 = prepare_matrix(json_pssm_1)
                    df2 = prepare_matrix(json_pssm_2)
                    df1 = normalise_matrix(df1)
                    df2 = normalise_matrix(df2)
                    print(df1)
                    print(df2)
                    pep_windows = calc_gini_windows(df1, df2)
                    index_1 = gini_window_index(df1)
                    # if len(index_1) == 0:
                    min_window = max(index_1) - min(index_1) + 1
                    print("min_window is ", min_window)
                    if min_window == 0:
                        min_window = min(pep_windows)
                        if min(pep_windows) > len(df1.columns):
                            min_window = len(df1.columns)
                    if min_window > len(df2.columns):
                        min_window = len(df2.columns)
                    index_2 = gini_window_index(df2)
                else:
                    pep_windows.append(int(pep_window))

                print("pep_windows: ", pep_windows)

                for window in pep_windows:
                    if window >= min_window:
                        res = {}
                        try:
                            base_pssm = json.dumps(data[pssm_i]["pssm"])
                            second_pssm = json.dumps(data[pssm_j]["pssm"])

                            (
                                equality,
                                _,
                                _,
                                ssd_global,
                                comparison_results,
                                df1,
                                df2,
                                ssd,
                                pearsons,
                                spearmans,
                                kendalls,
                                dot_products,
                                kl_divergence,
                                gini_a,
                                gini_b,
                            ) = compare_two_files(base_pssm, second_pssm, window)
                            res["equality"] = equality
                            res["base"] = data[pssm_i]["motif"]
                            res["second"] = data[pssm_j]["motif"]
                            res["ssd_global"] = ssd_global
                            res["comparison_results"] = comparison_results[0]
                            res["df1"] = df1
                            res["df2"] = df2
                            res["ssd"] = ssd
                            res["pearsons"] = pearsons
                            res["spearmans"] = spearmans
                            res["kendalls"] = kendalls
                            res["dot_products"] = dot_products
                            res["kl_divergence"] = kl_divergence
                            res["pep_window"] = window
                            res["norm_window"] = comparison_results[0][1] / window

                            row[data[pssm_j]["motif"]] = res["norm_window"]
                            results.append(res)

                        except TypeError as ex:
                            tqdm.write("error: {} on pssm: {}".format(ex, pssm_i))

                        except IndexError as ex:
                            tqdm.write(
                                "error: {} on pssm: {}, res: {} ".format(
                                    ex, pssm_i, res
                                )
                            )
                        except NoResultsError as ex:
                            tqdm.write(
                                "error: {} on pssm: {}, res: {} ".format(
                                    ex, pssm_i, res
                                )
                            )
                            continue

            df.loc[data[pssm_i]["motif"]] = row
            df.to_csv("ELMvsELM.csv")

        results.sort(key=lambda x: float(x["norm_window"]), reverse=True)
        # results.sort(key=lambda x: x["comparison_results"][1], reverse=True)
        return results


def print_df_ranges(
    df, region, ssd, pearsons, spearmans, kendalls, dot_products, kl_divergence
):
    """
    Prints the best regions, comparison wise of 2 PSSMs along with all their measures.
    """

    a, b = region.split(":")
    gini = gini_weight(df)
    gini = gini.T
    new_df = df.append(gini)
    new_df = new_df.rename(index={0: "Gini"})

    column_list = list(range(int(a), int(b) + 1))
    ssd.columns = column_list
    new_df = new_df.append(ssd)

    pearsons.columns = column_list
    new_df = new_df.append(pearsons)

    spearmans.columns = column_list
    new_df = new_df.append(spearmans)

    kendalls.columns = column_list
    new_df = new_df.append(kendalls)

    dot_products.columns = column_list
    new_df = new_df.append(dot_products)

    kl_divergence.columns = column_list
    new_df = new_df.append(kl_divergence)

    print(new_df.loc[:, int(a) : int(b)])

def Find_Optimal_Cutoff(target, predicted, label):
    """ 
    Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns: list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted, pos_label=label)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold']) 

def plot_important_positions(single_file):
    """
    Takes a single file with multiple PSSMs and calculates for each PSSM the number
    of important and unimportant positions. Then it creates a boxplot graph.
    """
    total_positions = 0
    important_positions = []
    unimportant_positions =[]

    with open(single_file) as json_file:
        data = json.load(json_file)

        for pssm in tqdm(data):
            try:
                json_pssm = json.dumps(data[pssm]["other_scoring_methods"]["log odds"])
                print("-----> ", data[pssm]["motif"])
                df = prepare_matrix(json_pssm)
                total_positions += len(df.columns)
                df_norm = normalise_matrix(df)
                gini = gini_weight(df_norm)
                position_list = data[pssm]["expanded_motif"]
                print("expanded motif ",position_list)

                position_index = [x for x in range(0, len(df.columns))]
                positions = dict(zip(position_index, position_list))

                for col,position in positions.items():
                    if position == ".":
                        unimportant_positions.append(gini.iloc[col,0])
                    else:
                        important_positions.append(gini.iloc[col,0])
            except TypeError as ex:
                            tqdm.write("error: {} on pssm: {}".format(ex, pssm))

            except IndexError as ex:
                tqdm.write(
                    "error: {} on pssm: {}".format(
                        ex, pssm
                    )
                )
            except NoResultsError as ex:
                tqdm.write(
                    "error: {} on pssm: {} ".format(
                        ex, pssm
                    )
                )
                continue
        important_df = pd.DataFrame(important_positions)
        important_df = important_df.rename(columns={0: "Important"})
        unimportant_df = pd.DataFrame(unimportant_positions)
        unimportant_df = unimportant_df.rename(columns={0: "Unimportant"})

        print("This is the min of the important: ", important_df.min())
        print("This is the max of the important: ", important_df.max())
        print("This is the length of the important: ", important_df.shape[0])
        print("This is the mean of the unimportant: ", unimportant_df.mean())
        comb_df = pd.concat([important_df, unimportant_df], axis=1) 
        print(comb_df)
        mdf = pd.melt(comb_df)
        print(mdf)
        mdf = mdf.rename(columns={"variable": "Positions", "value":"Gini"})

        ax = sns.boxplot(x="Positions", y="Gini", data=mdf)
        plt.show()

        # Calculate the optimal Cutoff and the area under the ROC curve
        mdf = mdf.dropna(axis=0)
        # mdf["Gini"] = 1-mdf["Gini"]
        print(mdf)
        fpr, tpr, thresholds = roc_curve(mdf["Positions"], mdf["Gini"], pos_label="Important", drop_intermediate = False)

        print("fpr ",fpr)
        print(len(fpr))
        print("tpr", tpr)
        print(len(tpr))
        print("thresholds ", thresholds)
        print(len(thresholds))
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve : ",roc_auc)

     

        threshold = Find_Optimal_Cutoff(mdf["Positions"], mdf["Gini"], "Important")
        print("This is the optimal cutoff: ",threshold)
        
        # # Plot of a ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic Curve')
        plt.legend(loc="lower right")
        plt.show()
        return ax.figure.savefig("boxplot_logodds.png")




