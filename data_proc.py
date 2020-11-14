import json
import math
import operator
from operator import getitem

from result import Result

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve
from tqdm.auto import tqdm

from utils import calculate_similarity, gini_weight, NoResultsError


def process_data(iters):
    """
    Compares two matrices using a window of comparison and returns a dictionary
    containing the positions of each matrix and the SSD
    """
    results = {}
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

        (_, pearsons_cor, _, _, _, _,) = calculate_similarity(a_df, b_df)

        comparison = pd.DataFrame(
            pearsons_cor.values * a_gini_df.values * b_gini_df.values,
            columns=pearsons_cor.columns,
            index=pearsons_cor.index,
        )

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

    return Result(
        comparison_results=results_sorted,
        similarity=pearsons_res,
        gini_1=ginis_1_res,
        gini_2=ginis_2_res,
        motif_1=motif_1,
        motif_2=motif_2,
    )


def calc_motif(df):
    motif = ""
    motif_list =[]
    cutoff = 0.57

    for col in df.columns:
        gini_df = gini_weight(df[col].to_frame())
        if gini_df.values[0][0] >= cutoff:
            ind = df[df[col] == df[col].max()].index.tolist()
            motif_list.append(ind[0])
        else:
            motif_list.append("x")


    return motif.join(motif_list)
