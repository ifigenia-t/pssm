import json
import math
import operator
from operator import getitem

import numpy as np
import pandas as pd
import math
from tqdm.auto import tqdm


pd.set_option("display.max_rows", 500)
pd.set_option("display.max_columns", 500)
pd.set_option("display.width", 1000)


class NoResultsError(Exception):
    pass


def prepare_matrix(filename):
    """
    Opens a json file and converts it to a dataframe of the correct format
    """
    df = pd.read_json(filename)
    df = df.dropna(axis=1, how="all")
    df_proc = df.T
    df_proc = df_proc.dropna()
    df_proc = df_proc.round(decimals=1)
    df_proc = df_proc.sort_index(axis=0)
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


# Importance Metric
def gini_weight(df):
    """
    Calculates the Gini Coefficient which is a measure of statistical dispersion.
    Gini = 1 means maximal inequallity
    Gini = 0 means perfect equallity where all the values are the same. 
    """
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
    pearson_scale = (pearson + 1) / 2
    pearson_scale = pearson_scale.round(decimals=3)

    return pearson_scale


def calc_spearmans_correlation(dfi, dfj):
    """
    Calculates the Spearman's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1. 
    Order is decreasing.
    """
    spearman = dfi.corr(dfj, method="spearman")
    spearman = round(spearman, 3)

    # Turning the correlation coefficient scale from -1 - 1 to 0-1
    spearmans_scale = (spearman + 1) / 2
    spearmans_scale = round(spearmans_scale, 3)

    return spearmans_scale


def calc_kendall_correlation(dfi, dfj):
    """
    Calculates Kendall's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1. 
    Order is decreasing.
    """
    kendall = dfi.corr(dfj, method="kendall")
    kendall = round(kendall, 3)

    # Turning the correlation coefficient scale from -1 - 1 to 0-1
    kendalls_scale = (kendall + 1) / 2
    kendalls_scale = round(kendalls_scale, 3)

    return kendalls_scale


def convert_to_df(data, tag):
    """
    Coverts data into dataframes with a given tag as an index name.
    """
    df = pd.DataFrame(data)
    df = df.T
    df = df.rename(index={0: tag})
    return df


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

        kendalls_scale = calc_kendall_correlation(dfi, dfj)
        pearsons_scale = calc_pearson_correlation(dfi, dfj)
        spearmans_scale = calc_spearmans_correlation(dfi, dfj)
        dot_product = calc_dot_product(dfi, dfj)
        ssd = calc_sum_of_squared_distance(dfi, dfj)
        kl = calc_Kullback_Leibler_distance(dfi, dfj)

        kendalls.append(kendalls_scale)
        pearsons.append(pearsons_scale)
        spearmans.append(spearmans_scale)
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

def get_df_window(df1, df2, pep_window, i, j):
    """
    Slices the dataframes according to a given window size.
    """
    a = df1.loc[:, i : i + pep_window -1]
    b = df2.loc[:, j : j + pep_window -1]

    a.columns = [ind for ind in range(0, len(a.columns))]
    b.columns = [ind for ind in range(0, len(b.columns))]

    return a, b

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

    it_window_a = len(df1.columns) - pep_window
    it_window_b = len(df2.columns) - pep_window

    for i in range(0, it_window_a+1):
        for j in range(0, it_window_b+1):
            key = "{}:{} - {}:{}".format(i, i + pep_window-1, j, j + pep_window-1)
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

    return (
        results_sorted,
        ssd_res,
        pearsons_res,
        spearmans_res,
        kendalls_res,
        dot_product_res,
        kl_divergence_res,
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
    )


def compare_combined_file(base_file, combined_file, pep_window):
    """
    Calculate all the comparisons for a PSSM and a .json file cointaing multiple
    PSSMs
    """

    with open(combined_file) as json_file:
        data = json.load(json_file)

        results = []

        i = 0

        for pssm in tqdm(data):
            res = {}
            try:
                json_pssm = json.dumps(data[pssm]["pssm"])
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
                ) = compare_two_files(base_file, json_pssm, pep_window)
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
                results.append(res)

            except TypeError as ex:
                print("error: {} on pssm: {}".format(ex, pssm))

            except IndexError as ex:
                print("error: {} on pssm: {}, res: {} ".format(ex, pssm, res))
            except NoResultsError as ex:
                print("error: {} on pssm: {}".format(ex, pssm))
                continue

        results.sort(key=lambda x: x["comparison_results"][1], reverse=True)

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

        i = 0

        for pssm_i in tqdm(data):

            row = {}

            for pssm_j in tqdm(data):
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
                    ) = compare_two_files(base_pssm, second_pssm, pep_window)
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

                    row[data[pssm_j]["motif"]] = comparison_results[0][1]
                    results.append(res)

                except TypeError as ex:
                    tqdm.write("error: {} on pssm: {}".format(ex, pssm_i))

                except IndexError as ex:
                    tqdm.write(
                        "error: {} on pssm: {}, res: {} ".format(ex, pssm_i, res)
                    )
                except NoResultsError as ex:
                    tqdm.write(
                        "error: {} on pssm: {}, res: {} ".format(ex, pssm_i, res)
                    )
                    continue

            df.loc[data[pssm_i]["motif"]] = row
            df.to_csv("res.csv")

        results.sort(key=lambda x: x["comparison_results"][1], reverse=True)
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
