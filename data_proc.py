import operator
from operator import getitem

from scipy.stats.stats import spearmanr

from result import Result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from tqdm.auto import tqdm

from data_prep import gini_weight, convert_to_df


class NoResultsError(Exception):
    pass


def process_data(iters, similarity_metric, multi_metrics=False, correct_results_file=""):
    """
    Compares two matrices using a window of comparison and returns a dictionary
    containing the positions of each matrix and the SSD
    """
    results = {}
    results_kendalls = {}
    results_spearmans = {}
    results_dots = {}
    results_ssds = {}
    results_kls = {}
    results_sorted = None
    pearsons = {}
    ginis_1 = {}
    ginis_2 = {}
    all_a_dfs = {}
    all_b_dfs = {}

    for i in iters:
        key = i[0]
        a_df = i[1]
        b_df = i[2]

        a_gini_df = gini_weight(a_df)
        a_gini_df = a_gini_df.T
        b_gini_df = gini_weight(b_df)
        b_gini_df = b_gini_df.T

        (kendalls, pearsons_cor, spearmans, dots, ssds, kls) = calculate_similarity(
            a_df, b_df, similarity_metric
        )

        # Main Comparison Calculation
        comparison = pd.DataFrame(
            pearsons_cor.values * a_gini_df.values * b_gini_df.values,
            columns=pearsons_cor.columns,
            index=pearsons_cor.index,
        )

        if multi_metrics:
            comparison_kendalls = pd.DataFrame(
                kendalls.values * a_gini_df.values * b_gini_df.values,
                columns=kendalls.columns,
                index=kendalls.index,
            )
            comparison_spearmans = pd.DataFrame(
                spearmans.values * a_gini_df.values * b_gini_df.values,
                columns=spearmans.columns,
                index=spearmans.index,
            )
            comparison_dots = pd.DataFrame(
                dots.values * a_gini_df.values * b_gini_df.values,
                columns=dots.columns,
                index=dots.index,
            )
            comparison_ssds = pd.DataFrame(
                ssds.values * a_gini_df.values * b_gini_df.values,
                columns=ssds.columns,
                index=ssds.index,
            )
            comparison_kls = pd.DataFrame(
                kls.values * a_gini_df.values * b_gini_df.values,
                columns=kls.columns,
                index=kls.index,
            )
            results_kendalls[key] = comparison_kendalls.values.sum()
            results_spearmans[key] = comparison_spearmans.values.sum()
            results_dots[key] = comparison_dots.values.sum()
            results_ssds[key] = comparison_ssds.values.sum()
            results_kls[key] = comparison_kls.values.sum()

        results[key] = comparison.values.sum()
        pearsons[key] = pearsons_cor
        ginis_1[key] = a_gini_df
        ginis_2[key] = b_gini_df
        all_a_dfs[key] = a_df
        all_b_dfs[key] = b_df

    results_sorted = sorted(results.items(), key=operator.itemgetter(1), reverse=True)

    if results_sorted is None:
        raise NoResultsError("No result")

    res_0 = results_sorted[0]

    pearsons_res = pearsons[res_0[0]]
    ginis_1_res = ginis_1[res_0[0]]
    ginis_2_res = ginis_2[res_0[0]]

    motif_1 = calc_motif(all_a_dfs[res_0[0]])
    motif_2 = calc_motif(all_b_dfs[res_0[0]])

    res = Result(
        comparison_results=results_sorted,
        similarity=pearsons_res,
        gini_1=ginis_1_res,
        gini_2=ginis_2_res,
        motif_1=motif_1,
        motif_2=motif_2,
    )

    if multi_metrics:
        res.comparison_results_kendalls = sorted(
            results_kendalls.items(), key=operator.itemgetter(1), reverse=True
        )
        res.comparison_results_spearmans = sorted(
            results_spearmans.items(), key=operator.itemgetter(1), reverse=True
        )
        res.comparison_results_dots = sorted(
            results_dots.items(), key=operator.itemgetter(1)
        )
        res.comparison_results_ssds = sorted(
            results_ssds.items(), key=operator.itemgetter(1)
        )
        res.comparison_results_kls = sorted(
            results_kls.items(), key=operator.itemgetter(1)
        )

    return res

def find_rank_name(elms, pssm, metric, reverse=True):
    ranks = []
    elms[metric].sort(key=operator.itemgetter("result"), reverse=reverse)
    for i, e in enumerate(elms[metric]):
        if e["elm"] == pssm:
            ranks.append(i)

    return ranks

def calc_motif(df):
    motif = ""
    motif_list = []
    cutoff = 0.58
    gini_df = gini_weight(df)
    gini_mean = gini_df.mean()
    mean_cutoff = gini_mean.mean()

    for col in df.columns:
        gini_df = gini_weight(df[col].to_frame())
        if gini_df.values[0][0] >= cutoff:
            ind = df[df[col] == df[col].max()].index.tolist()
            motif_list.append(ind[0])
        elif gini_df.values[0][0] >= mean_cutoff:
            ind = df[df[col] == df[col].max()].index.tolist()
            motif_list.append(ind[0])
        else:
            motif_list.append("x")

    return motif.join(motif_list)


def calculate_similarity(df1, df2, similarity_metric):
    """
    Calculates all the similarity measures column wise and returns 1 row dataframes.
    """
    kendalls = []
    pearsons = []
    spearmans = []
    dots = []
    ssds = []
    kls = []
    kendalls_df = None
    pearsons_df = None
    spearmans_df = None
    dots_df = None
    ssds_df = None
    kls_df = None

    for i in range(0, len(df1.columns)):
        dfi = df1.iloc[:, i]
        dfj = df2.iloc[:, i]

        if "kendall" in similarity_metric:
            kendall = calc_kendall_correlation(dfi, dfj)
            kendalls.append(kendall)

        if "pearson" in similarity_metric:
            pearson = calc_pearson_correlation(dfi, dfj)
            pearsons.append(pearson)

        if "spearman" in similarity_metric:
            spearman = calc_spearmans_correlation(dfi, dfj)
            spearmans.append(spearman)

        if "dot" in similarity_metric:
            dot_product = calc_dot_product(dfi, dfj)
            dots.append(dot_product)
        if "ssd" in similarity_metric:
            ssd = calc_sum_of_squared_distance(dfi, dfj)
            ssds.append(ssd)
        if "kl" in similarity_metric:
            kl = calc_Kullback_Leibler_distance(dfi, dfj)
            kls.append(kl)

    if "kendall" in similarity_metric:
        kendalls_df = convert_to_df(kendalls, "Kendall")
    if "pearson" in similarity_metric:
        pearsons_df = convert_to_df(pearsons, "Pearson")
    if "spearman" in similarity_metric:
        spearmans_df = convert_to_df(spearmans, "Spearman")
    if "dot" in similarity_metric:
        dots_df = convert_to_df(dots, "Dot")
    if "ssd" in similarity_metric:
        ssds_df = convert_to_df(ssds, "SSD")
    if "kl" in similarity_metric:
        kls_df = convert_to_df(kls, "KL")

    return kendalls_df, pearsons_df, spearmans_df, dots_df, ssds_df, kls_df


def calc_kendall_correlation(dfi, dfj):
    """
    Calculates Kendall's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1.
    Order is decreasing.
    """
    kendall = dfi.corr(dfj, method="kendall")
    # kendall = round(kendall, 3)

    return kendall


def calc_pearson_correlation(dfi, dfj):
    """
    Calculates Pearson's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1.
    Order is decreasing.
    """
    pearson = dfi.corr(dfj)
    # pearson = pearson.round(decimals=3)

    return pearson


def calc_spearmans_correlation(dfi, dfj):
    """
    Calculates the Spearman's correlation between two dataframes and rescales them
    from -1 - 1 to 0-1.
    Order is decreasing.
    """
    spearman = dfi.corr(dfj, method="spearman")
    # spearman = round(spearman, 3)

    return spearman


def calc_dot_product(dfi, dfj):
    """
    Calculates the dot product between 2 dataframes and returns their sum.
    Order is ascending.
    """
    dot_product = dfi.values * dfj.values
    dot_product_sum = dot_product.sum() #.round(decimals=3)

    return dot_product_sum


def calc_sum_of_squared_distance(dfi, dfj):
    """
    Calculates the square distance between 2 dataframes and returns their sum.
    Order is ascending.
    """
    ssd = (dfi - dfj) ** 2
    ssd_sum = ssd.sum() #.round(decimals=3)

    return ssd_sum


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