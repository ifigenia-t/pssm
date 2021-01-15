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

from utils import calculate_similarity, gini_weight, NoResultsError


def process_data(iters, multi_metrics=False):
    """
    Compares two matrices using a window of comparison and returns a dictionary
    containing the positions of each matrix and the SSD
    """
    results = {}
    results_kendalls = {}
    results_spearmans ={}
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

        # print("calculating simliarity")
        (kendalls, pearsons_cor, spearmans, dots, ssds, kls) = calculate_similarity(a_df, b_df)

        # print("done")
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
            results_kendalls[key] = comparison_kendalls.values.sum().round(decimals=3)
            results_spearmans[key] = comparison_spearmans.values.sum().round(decimals=3)
            results_dots[key] = comparison_dots.values.sum().round(decimals=3) 
            results_ssds[key] = comparison_ssds.values.sum().round(decimals=3) 
            results_kls[key] = comparison_kls.values.sum().round(decimals=3)




        results[key] = comparison.values.sum().round(decimals=3)
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
        res.comparison_results_kendalls = sorted(results_kendalls.items(), key=operator.itemgetter(1), reverse=True)
        res.comparison_results_spearmans = sorted(results_spearmans.items(), key=operator.itemgetter(1), reverse=True)
        res.comparison_results_dots = sorted(results_dots.items(), key=operator.itemgetter(1))
        res.comparison_results_ssds = sorted(results_ssds.items(), key=operator.itemgetter(1))
        res.comparison_results_kls = sorted(results_kls.items(), key=operator.itemgetter(1))

    return res


def calc_motif(df):
    motif = ""
    motif_list =[]
    cutoff = 0.58
    gini_df = gini_weight(df)
    gini_mean = gini_df.mean()
    mean_cutoff = gini_mean.mean()

    for col in df.columns:
        gini_df = gini_weight(df[col].to_frame())
        if gini_df.values[0][0] >= cutoff:
            ind = df[df[col] == df[col].max()].index.tolist()
            motif_list.append(ind[0])
        # elif df[col].max() >= mean_cutoff:
        elif gini_df.values[0][0] >= mean_cutoff:
            ind = df[df[col] == df[col].max()].index.tolist()
            motif_list.append(ind[0])
        else:
            motif_list.append("x")


    return motif.join(motif_list)
