import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pandas.core.indexes import base
from scipy import stats
from sklearn.metrics import auc, roc_auc_score, roc_curve
from tqdm.auto import tqdm

from data_prep import gini_weight, normalise_matrix, prepare_matrix


class NoResultsError(Exception):
    pass


def convert_df_to_string(df):
    """
    Takes a dataframe and converts it to a list and then to a sting separated
    by ';'
    """
    if len(df.columns) > 1:
        df = df.T
        df = df.rename({df.columns[0]: 0}, axis=1)
    else:
        df = df.round(decimals=3)
    df_list = df[0].values.tolist()
    df_str = ";".join(map(str, df_list))
    return df_str


def output_data(results):
    res_best = results[0]
    print(res_best)


def find_rank_file(elms, correct_res, metric, reverse=True):
    ranks = []
    elms[metric].sort(key=operator.itemgetter("result"), reverse=reverse)
    for cr in correct_res:
        for i, e in enumerate(elms[metric]):
            if e["elm"] == cr:
                ranks.append(i)

    return ranks


def find_rank_name(elms, pssm, metric, reverse=True):
    ranks = []
    elms[metric].sort(key=operator.itemgetter("result"), reverse=reverse)
    for i, e in enumerate(elms[metric]):
        if e["elm"] == pssm:
            ranks.append(i)

    return ranks


class MultiComparison:
    def __init__(
        self,
        data_file_1,
        data_file_2="",
        file_name="",
        use_index=True,
        correct_results_file="",
    ):
        self.multi_metrics = {
            "pearsons": {"match": [], "mismatch": []},
            "kendalls": {"match": [], "mismatch": []},
            "spearmans": {"match": [], "mismatch": []},
            "dots": {"match": [], "mismatch": []},
            "ssds": {"match": [], "mismatch": []},
            "kls": {"match": [], "mismatch": []},
        }

        self.best_match = {}
        self.all_ranks = {}

        self.pearson_rank = []
        self.kendall_rank = []
        self.spearman_rank = []
        self.ssd_rank = []
        self.dot_rank = []
        self.kl_rank = []

        if use_index:
            data_1_names = []
            for pssm in data_file_1:
                data_1_names.append(data_file_1[pssm]["motif"])
        else:
            data_1_names = [file_name]

        col_names = [
            "ELM",
            "Quality",
            "Comparison Results",
            "Motif 1",
            "Motif 2",
            "Consensus",
            "Comparison Windows",
            "Gini 1",
            "Gini 2",
            "Similarity",
        ]
        similarity_met_col_names = [
            "Pearsons",
            "Kendalls",
            "Spearmans",
            "SSD",
            "Dot Product",
            "KL",
        ]

        self.best_df = pd.DataFrame(columns=col_names, index=data_1_names)
        self.similarity_df = pd.DataFrame(
            columns=similarity_met_col_names, index=data_1_names
        )

        if data_file_2 != "":
            data_2_names = []
            for pssm in data_file_2:
                data_2_names.append(data_file_2[pssm]["motif"])
            self.all_df = pd.DataFrame(columns=data_1_names, index=data_2_names)

        if correct_results_file != "":
            with open(correct_results_file) as crf:
                self.correct_results = json.load(crf)
                self.match = []
                self.mismatch = []

    def create_file(self):
        self.best_df = self.best_df.sort_values(
            by=["Comparison Results"], ascending=False
        )
        # TODO: use filenames
        self.best_df.to_csv("elm-best.csv")
        if hasattr(self, "all_df"):
            self.all_df.to_csv("elm-all.csv")
        if hasattr(self, "similarity_df"):
            print(self.similarity_df)
            self.similarity_df.to_csv("proppd_similarity.csv")

    def plot_match(self):
        match_df = pd.DataFrame(self.match, columns=["Match"])
        mismatch_df = pd.DataFrame(self.mismatch, columns=["Mismatch"])

        comb_df = pd.concat([match_df, mismatch_df], axis=1)
        mdf = pd.melt(comb_df)
        mdf = mdf.rename(columns={"variable": "Identification", "value": "Comparison"})
        ax = sns.boxplot(x="Identification", y="Comparison", data=mdf)

        plt.show()

    def plot_ROC(self):
        match_df = pd.DataFrame(self.match, columns=["Match"])
        match_df = match_df.rename(columns={0: "Match"})
        mismatch_df = pd.DataFrame(self.mismatch, columns=["Mismatch"])
        mismatch_df = mismatch_df.rename(columns={0: "Mismatch"})

        comb_df = pd.concat([match_df, mismatch_df], axis=1)

        mdf = pd.melt(comb_df)
        mdf = mdf.rename(columns={"variable": "Identification", "value": "Comparison"})
        mdf = mdf.dropna(axis=0)
        # mdf["Comparison"] = 1-mdf["Comparison"]
        fpr, tpr, thresholds = roc_curve(
            mdf["Identification"], mdf["Comparison"], pos_label="Match"
        )

        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            label="ROC curve (area = %0.2f)" % auc(fpr, tpr),
        )
        plt.plot(color="navy", linestyle="--")
        plt.title("ROC curve")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.legend(loc="lower right")
        plt.show()

        # ROC Curve
        # print("This is the AUC ",roc_auc_score(comb_df["Match"], comb_df["Mismatch"]))

    def add(self, result):
        print("add")
        if (
            result.comparison_results[0][1]
            > self.best_df.at[result.base_name, "Comparison Results"]
        ) or pd.isna(self.best_df.at[result.base_name, "Comparison Results"]):
            print("adding...")
            self.best_df.loc[result.base_name] = [
                result.elm,
                result.quality,
                result.comparison_results[0][1],
                result.motif_1,
                result.motif_2,
                result.consensus,
                result.comparison_results[0][0],
                convert_df_to_string(result.gini_1),
                convert_df_to_string(result.gini_2),
                convert_df_to_string(result.similarity),
            ]

        if hasattr(self, "correct_results"):
            print("comparing")
            if result.elm in self.correct_results[result.base_name]:
                print(result.base_name, result.elm, "match")
                self.match.append(result.comparison_results[0][1])
                print("match", result.comparison_results[1])
            else:
                self.mismatch.append(result.comparison_results[0][1])
                print(result.base_name, result.elm, "miss")

        if hasattr(self, "all_df"):
            print("adding 2...", result.elm)
            self.all_df.at[result.elm, result.base_name] = result.comparison_results[0][
                1
            ]

        pearson_comp = []

    def add_multi_metrics_file(self, result):
        """
        Checks the correct results based on a correct results map.
        Then it appends the result in the respective match/mismatch lists
        for every simmilarity metric.
        """
        if result.elm in self.correct_results[result.base_name]:
            self.multi_metrics["pearsons"]["match"].append(
                result.comparison_results[0][1]
            )
            self.multi_metrics["kendalls"]["match"].append(
                result.comparison_results_kendalls[0][1]
            )
            self.multi_metrics["spearmans"]["match"].append(
                result.comparison_results_spearmans[0][1]
            )
            self.multi_metrics["dots"]["match"].append(
                result.comparison_results_dots[0][1]
            )
            self.multi_metrics["ssds"]["match"].append(
                result.comparison_results_ssds[0][1]
            )
            self.multi_metrics["kls"]["match"].append(
                result.comparison_results_kls[0][1]
            )
        else:
            self.multi_metrics["pearsons"]["mismatch"].append(
                result.comparison_results[0][1]
            )
            self.multi_metrics["kendalls"]["mismatch"].append(
                result.comparison_results_kendalls[0][1]
            )
            self.multi_metrics["spearmans"]["mismatch"].append(
                result.comparison_results_spearmans[0][1]
            )
            self.multi_metrics["dots"]["mismatch"].append(
                result.comparison_results_dots[0][1]
            )
            self.multi_metrics["ssds"]["mismatch"].append(
                result.comparison_results_ssds[0][1]
            )
            self.multi_metrics["kls"]["mismatch"].append(
                result.comparison_results_kls[0][1]
            )

        if result.base_name not in self.best_match:
            self.best_match[result.base_name] = {
                "pearsons": {},
                "kendalls": {},
                "spearmans": {},
                "dots": {},
                "ssds": {},
                "kls": {},
            }
        if result.base_name not in self.all_ranks:
            self.all_ranks[result.base_name] = {
                "pearsons": [],
                "kendalls": [],
                "spearmans": [],
                "dots": [],
                "ssds": [],
                "kls": [],
            }

        # pearsons
        if self.best_match[result.base_name]["pearsons"] == {}:
            self.best_match[result.base_name]["pearsons"] = {
                "elm": result.elm,
                "result": result.comparison_results[0][1],
            }
        else:
            br = self.best_match[result.base_name]["pearsons"]["result"]
            if br < result.comparison_results[0][1]:
                self.best_match[result.base_name]["pearsons"] = {
                    "elm": result.elm,
                    "result": result.comparison_results[0][1],
                }
        self.all_ranks[result.base_name]["pearsons"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results[0][1],
            }
        )

        # kendalls
        if self.best_match[result.base_name]["kendalls"] == {}:
            self.best_match[result.base_name]["kendalls"] = {
                "elm": result.elm,
                "result": result.comparison_results_kendalls[0][1],
            }
        else:
            br = self.best_match[result.base_name]["kendalls"]["result"]
            if br < result.comparison_results_kendalls[0][1]:
                self.best_match[result.base_name]["kendalls"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_kendalls[0][1],
                }
        self.all_ranks[result.base_name]["kendalls"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_kendalls[0][1],
            }
        )

        # spearmans
        if self.best_match[result.base_name]["spearmans"] == {}:
            self.best_match[result.base_name]["spearmans"] = {
                "elm": result.elm,
                "result": result.comparison_results_spearmans[0][1],
            }
        else:
            br = self.best_match[result.base_name]["spearmans"]["result"]
            if br < result.comparison_results_spearmans[0][1]:
                self.best_match[result.base_name]["spearmans"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_spearmans[0][1],
                }
        self.all_ranks[result.base_name]["spearmans"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_spearmans[0][1],
            }
        )

        # dots
        if self.best_match[result.base_name]["dots"] == {}:
            self.best_match[result.base_name]["dots"] = {
                "elm": result.elm,
                "result": result.comparison_results_dots[0][1],
            }
        else:
            br = self.best_match[result.base_name]["dots"]["result"]
            if br < result.comparison_results_dots[0][1]:
                self.best_match[result.base_name]["dots"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_dots[0][1],
                }
        self.all_ranks[result.base_name]["dots"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_dots[0][1],
            }
        )

        # ssds
        if self.best_match[result.base_name]["ssds"] == {}:
            self.best_match[result.base_name]["ssds"] = {
                "elm": result.elm,
                "result": result.comparison_results_ssds[0][1],
            }
        else:
            br = self.best_match[result.base_name]["ssds"]["result"]
            if br < result.comparison_results_ssds[0][1]:
                self.best_match[result.base_name]["ssds"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_ssds[0][1],
                }
        self.all_ranks[result.base_name]["ssds"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_ssds[0][1],
            }
        )

        # kls
        if self.best_match[result.base_name]["kls"] == {}:
            self.best_match[result.base_name]["kls"] = {
                "elm": result.elm,
                "result": result.comparison_results_kls[0][1],
            }
        else:
            br = self.best_match[result.base_name]["kls"]["result"]
            if br < result.comparison_results_kls[0][1]:
                self.best_match[result.base_name]["kls"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_kls[0][1],
                }
        self.all_ranks[result.base_name]["kls"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_kls[0][1],
            }
        )

    def add_multi_metrics(self, result):
        """
        A result is considered correct when the predicted ELM class returned from each similarity metric
        is the same as the examined ELM class. (A match is a when the best result is itself)
        Subsequently, it appends the result in the respective match/mismatch lists
        for every simmilarity metric.
        """

        if result.base_name == result.elm:
            self.multi_metrics["pearsons"]["match"].append(
                result.comparison_results[0][1]
            )
            self.multi_metrics["kendalls"]["match"].append(
                result.comparison_results_kendalls[0][1]
            )
            self.multi_metrics["spearmans"]["match"].append(
                result.comparison_results_spearmans[0][1]
            )
            self.multi_metrics["dots"]["match"].append(
                result.comparison_results_dots[0][1]
            )
            self.multi_metrics["ssds"]["match"].append(
                result.comparison_results_ssds[0][1]
            )
            self.multi_metrics["kls"]["match"].append(
                result.comparison_results_kls[0][1]
            )

        else:
            self.multi_metrics["pearsons"]["mismatch"].append(
                result.comparison_results[0][1]
            )
            self.multi_metrics["kendalls"]["mismatch"].append(
                result.comparison_results_kendalls[0][1]
            )
            self.multi_metrics["spearmans"]["mismatch"].append(
                result.comparison_results_spearmans[0][1]
            )
            self.multi_metrics["dots"]["mismatch"].append(
                result.comparison_results_dots[0][1]
            )
            self.multi_metrics["ssds"]["mismatch"].append(
                result.comparison_results_ssds[0][1]
            )
            self.multi_metrics["kls"]["mismatch"].append(
                result.comparison_results_kls[0][1]
            )

        if result.base_name not in self.best_match:
            self.best_match[result.base_name] = {
                "pearsons": {},
                "kendalls": {},
                "spearmans": {},
                "dots": {},
                "ssds": {},
                "kls": {},
            }

        if result.base_name not in self.all_ranks:
            self.all_ranks[result.base_name] = {
                "pearsons": [],
                "kendalls": [],
                "spearmans": [],
                "dots": [],
                "ssds": [],
                "kls": [],
            }

        # pearsons
        if self.best_match[result.base_name]["pearsons"] == {}:
            self.best_match[result.base_name]["pearsons"] = {
                "elm": result.elm,
                "result": result.comparison_results[0][1],
            }
        else:
            br = self.best_match[result.base_name]["pearsons"]["result"]
            if br < result.comparison_results[0][1]:
                self.best_match[result.base_name]["pearsons"] = {
                    "elm": result.elm,
                    "result": result.comparison_results[0][1],
                }

        self.all_ranks[result.base_name]["pearsons"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results[0][1],
            }
        )

        # kendalls
        if self.best_match[result.base_name]["kendalls"] == {}:
            self.best_match[result.base_name]["kendalls"] = {
                "elm": result.elm,
                "result": result.comparison_results_kendalls[0][1],
            }
        else:
            br = self.best_match[result.base_name]["kendalls"]["result"]
            if br < result.comparison_results_kendalls[0][1]:
                self.best_match[result.base_name]["kendalls"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_kendalls[0][1],
                }
        self.all_ranks[result.base_name]["kendalls"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_kendalls[0][1],
            }
        )

        # spearmans
        if self.best_match[result.base_name]["spearmans"] == {}:
            self.best_match[result.base_name]["spearmans"] = {
                "elm": result.elm,
                "result": result.comparison_results_spearmans[0][1],
            }
        else:
            br = self.best_match[result.base_name]["spearmans"]["result"]
            if br < result.comparison_results_spearmans[0][1]:
                self.best_match[result.base_name]["spearmans"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_spearmans[0][1],
                }
        self.all_ranks[result.base_name]["spearmans"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_spearmans[0][1],
            }
        )

        # dots
        if self.best_match[result.base_name]["dots"] == {}:
            self.best_match[result.base_name]["dots"] = {
                "elm": result.elm,
                "result": result.comparison_results_dots[0][1],
            }
        else:
            br = self.best_match[result.base_name]["dots"]["result"]
            if br < result.comparison_results_dots[0][1]:
                self.best_match[result.base_name]["dots"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_dots[0][1],
                }
        self.all_ranks[result.base_name]["dots"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_dots[0][1],
            }
        )

        # ssds
        if self.best_match[result.base_name]["ssds"] == {}:
            self.best_match[result.base_name]["ssds"] = {
                "elm": result.elm,
                "result": result.comparison_results_ssds[0][1],
            }
        else:
            br = self.best_match[result.base_name]["ssds"]["result"]
            if br < result.comparison_results_ssds[0][1]:
                self.best_match[result.base_name]["ssds"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_ssds[0][1],
                }
        self.all_ranks[result.base_name]["ssds"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_ssds[0][1],
            }
        )

        # kls
        if self.best_match[result.base_name]["kls"] == {}:
            self.best_match[result.base_name]["kls"] = {
                "elm": result.elm,
                "result": result.comparison_results_kls[0][1],
            }
        else:
            br = self.best_match[result.base_name]["kls"]["result"]
            if br < result.comparison_results_kls[0][1]:
                self.best_match[result.base_name]["kls"] = {
                    "elm": result.elm,
                    "result": result.comparison_results_kls[0][1],
                }
        self.all_ranks[result.base_name]["kls"].append(
            {
                "elm": result.elm,
                "result": result.comparison_results_kls[0][1],
            }
        )

    def mann_whitney_u_test(self):
        """
        Perform the Mann-Whitney U Test, comparing two different distributions.
        Args:
        distribution_1: List.
        distribution_2: List.
        Outputs:
            u_statistic: Float. U statisitic for the test.
            p_value: Float.
        """
        u_statistic, p_value = stats.mannwhitneyu(self.match, self.mismatch)
        return u_statistic, p_value

    def plot_multi_best_match_file(self):
        pearsons_match = 0
        kendalls_match = 0
        spearmans_match = 0
        dots_match = 0
        ssds_match = 0
        kls_match = 0

        if not hasattr(self, "correct_results"):
            raise Exception("should have correct file")

        for pssm, data in self.best_match.items():
            if data["pearsons"]["elm"] in self.correct_results[pssm]:
                pearsons_match += 1

            if data["kendalls"]["elm"] in self.correct_results[pssm]:
                kendalls_match += 1

            if data["spearmans"]["elm"] in self.correct_results[pssm]:
                spearmans_match += 1

            if data["dots"]["elm"] in self.correct_results[pssm]:
                dots_match += 1

            if data["ssds"]["elm"] in self.correct_results[pssm]:
                ssds_match += 1

            if data["kls"]["elm"] in self.correct_results[pssm]:
                kls_match += 1

        objects = (
            "Pearson",
            "Kendall",
            "Spearman",
            "Dot",
            "SSD",
            "KL",
        )

        y_pos = np.arange(len(objects))
        performance = [
            pearsons_match,
            kendalls_match,
            spearmans_match,
            dots_match,
            ssds_match,
            kls_match,
        ]

        print(json.dumps(self.best_match))

        fig, ax = plt.subplots()
        rects = ax.bar(
            y_pos,
            performance,
            align="center",
            alpha=0.5,
            color=["cyan", "orange", "green", "red", "purple", "brown"],
        )
        plt.xticks(y_pos, objects)

        plt.ylabel("No of correct Matches")
        plt.title("Benchmarking Similarity Metrics ProP-PD / ELM")
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                height,
                ha="center",
                va="bottom",
            )
        plt.show()

    def plot_multi_best_match(self):
        pearsons_match = 0
        kendalls_match = 0
        spearmans_match = 0
        dots_match = 0
        ssds_match = 0
        kls_match = 0

        print("\nself.best_match final\n", json.dumps(self.best_match), "\n")

        for pssm, data in self.best_match.items():
            if data["pearsons"]["elm"] == pssm:
                pearsons_match += 1

            if data["kendalls"]["elm"] == pssm:
                kendalls_match += 1

            if data["spearmans"]["elm"] == pssm:
                spearmans_match += 1

            if data["dots"]["elm"] == pssm:
                dots_match += 1

            if data["ssds"]["elm"] == pssm:
                ssds_match += 1

            if data["kls"]["elm"] == pssm:
                kls_match += 1

        objects = (
            "Pearson",
            "Kendall",
            "Spearman",
            "Dot",
            "SSD",
            "KL",
        )

        y_pos = np.arange(len(objects))
        performance = [
            pearsons_match,
            kendalls_match,
            spearmans_match,
            dots_match,
            ssds_match,
            kls_match,
        ]
        fig, ax = plt.subplots()
        rects = ax.bar(
            y_pos,
            performance,
            align="center",
            alpha=0.5,
            color=["cyan", "orange", "green", "red", "purple", "brown"],
        )
        plt.xticks(y_pos, objects)
        # for index, value in enumerate(performance):
        #     plt.text(value, index, str(value))
        plt.ylabel("No of correct Matches")
        plt.title("Benchmarking Similarity Metrics")
        for rect in rects:
            height = rect.get_height()
            ax.text(
                rect.get_x() + rect.get_width() / 2.0,
                height,
                height,
                ha="center",
                va="bottom",
            )
        plt.show()

    def plot_rank_boxplots(self):
        for pssm, data in self.best_match.items():
            if hasattr(self, "similarity_df"):
                print("adding similarity best", pssm)

                self.similarity_df.at[pssm] = [
                    data["pearsons"]["elm"],
                    data["kendalls"]["elm"],
                    data["spearmans"]["elm"],
                    data["ssds"]["elm"],
                    data["dots"]["elm"],
                    data["kls"]["elm"],
                ]

        if self.correct_results != "":
            for pssm, _ in self.all_ranks.items():
                self.pearson_rank.extend(
                    find_rank_file(
                        self.all_ranks[pssm], self.correct_results[pssm], "pearsons"
                    )
                )
                self.kendall_rank.extend(
                    find_rank_file(
                        self.all_ranks[pssm], self.correct_results[pssm], "kendalls"
                    )
                )
                self.spearman_rank.extend(
                    find_rank_file(
                        self.all_ranks[pssm], self.correct_results[pssm], "spearmans"
                    )
                )
                self.ssd_rank.extend(
                    find_rank_file(
                        self.all_ranks[pssm],
                        self.correct_results[pssm],
                        "ssds",
                        reverse=False,
                    )
                )
                self.dot_rank.extend(
                    find_rank_file(
                        self.all_ranks[pssm],
                        self.correct_results[pssm],
                        "dots",
                        reverse=False,
                    )
                )
                self.kl_rank.extend(
                    find_rank_file(
                        self.all_ranks[pssm],
                        self.correct_results[pssm],
                        "kls",
                        reverse=False,
                    )
                )

        else:
            for pssm, _ in self.all_ranks.items():
                self.pearson_rank.extend(
                    find_rank_name(self.all_ranks[pssm], pssm, "pearsons")
                )
                self.kendall_rank.extend(
                    find_rank_name(self.all_ranks[pssm], pssm, "kendalls")
                )
                self.spearman_rank.extend(
                    find_rank_name(self.all_ranks[pssm], pssm, "spearmans")
                )
                self.ssd_rank.extend(
                    find_rank_name(self.all_ranks[pssm], pssm, "ssds", reverse=False)
                )
                self.dot_rank.extend(
                    find_rank_name(self.all_ranks[pssm], pssm, "dots", reverse=False)
                )
                self.kl_rank.extend(
                    find_rank_name(self.all_ranks[pssm], pssm, "kls", reverse=False)
                )

        print("\nsorted new self.all_ranks final\n", json.dumps(self.all_ranks), "\n")
        print("\nPearsons Rank ", self.pearson_rank)
        print("Kendalls Rank ", self.kendall_rank)
        print("Spearmans Rank ", self.spearman_rank)
        print("SSDs Rank ", self.ssd_rank)
        print("Dot Rank ", self.dot_rank)
        print("KL Rank ", self.kl_rank)

        pearson_rank_df = pd.DataFrame(self.pearson_rank, columns=["Pearson"])
        kendall_rank_df = pd.DataFrame(self.kendall_rank, columns=["Kendall"])
        spearman_rank_df = pd.DataFrame(self.spearman_rank, columns=["Spearman"])
        ssd_rank_df = pd.DataFrame(self.ssd_rank, columns=["SSD"])
        dot_rank_df = pd.DataFrame(self.dot_rank, columns=["Dot"])
        kl_rank_df = pd.DataFrame(self.kl_rank, columns=["KL"])

        comb_df = pd.concat(
            [
                pearson_rank_df,
                kendall_rank_df,
                spearman_rank_df,
                ssd_rank_df,
                dot_rank_df,
                kl_rank_df,
            ],
            axis=1,
        )
        mdf = pd.melt(comb_df)
        mdf = mdf.rename(
            columns={"variable": "Similarity Metric", "value": "Correct Result Rank"}
        )
        my_pal = color = ["cyan", "orange", "green", "red", "purple", "brown"]
        ax = sns.boxplot(
            x="Similarity Metric", y="Correct Result Rank", data=mdf, palette=my_pal
        )

        plt.show()

    def plot_multi_roc(self):
        print("\n", json.dumps(self.multi_metrics))

        plt.figure()

        for metric, data in self.multi_metrics.items():

            df_match_name = metric + "_match_df"
            df_mismatch_name = metric + "mismatch_df"
            df_match_name = pd.DataFrame(data["match"], columns=["Match"])
            df_match_name = df_match_name.rename(columns={0: "Match"})
            df_mismatch_name = pd.DataFrame(data["mismatch"], columns=["Mismatch"])
            df_mismatch_name = df_mismatch_name.rename(columns={0: "Mismatch"})

            comb_df = pd.concat([df_match_name, df_mismatch_name], axis=1)
            mdf_name = "mdf_" + metric
            mdf_name = pd.melt(comb_df)
            mdf_name = mdf_name.rename(
                columns={"variable": "Identification", "value": "Comparison"}
            )
            mdf_name = mdf_name.dropna(axis=0)

            fpr, tpr, thresholds = roc_curve(
                mdf_name["Identification"], mdf_name["Comparison"], pos_label="Match"
            )
            plt.plot(fpr, tpr, label="{} (AUC = {})".format(metric, auc(fpr, tpr)))
            plt.plot(color="navy", linestyle="--")
            plt.title("ROC curve ELM")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend(loc="lower right")

        plt.show()


def plot_important_positions(single_file):
    """
    Takes a single file with multiple PSSMs and calculates for each PSSM the number
    of important and unimportant positions. Then it creates a boxplot graph.
    """
    total_positions = 0
    important_positions = []
    unimportant_positions = []

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
                print("expanded motif ", position_list)

                position_index = [x for x in range(0, len(df.columns))]
                positions = dict(zip(position_index, position_list))

                for col, position in positions.items():
                    if position == ".":
                        unimportant_positions.append(gini.iloc[col, 0])
                    else:
                        important_positions.append(gini.iloc[col, 0])
            except TypeError as ex:
                tqdm.write("error: {} on pssm: {}".format(ex, pssm))

            except IndexError as ex:
                tqdm.write("error: {} on pssm: {}".format(ex, pssm))
            except NoResultsError as ex:
                tqdm.write("error: {} on pssm: {} ".format(ex, pssm))
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
        mdf = mdf.rename(columns={"variable": "Positions", "value": "Gini"})

        ax = sns.boxplot(x="Positions", y="Gini", data=mdf)
        plt.show()

        # Calculate the optimal Cutoff and the area under the ROC curve
        mdf = mdf.dropna(axis=0)
        # mdf["Gini"] = 1-mdf["Gini"]
        print(mdf)
        fpr, tpr, thresholds = roc_curve(
            mdf["Positions"],
            mdf["Gini"],
            pos_label="Important",
            drop_intermediate=False,
        )

        print("fpr ", fpr)
        print(len(fpr))
        print("tpr", tpr)
        print(len(tpr))
        print("thresholds ", thresholds)
        print(len(thresholds))
        roc_auc = auc(fpr, tpr)
        print("Area under the ROC curve : ", roc_auc)

        threshold = find_optimal_cutoff(mdf["Positions"], mdf["Gini"], "Important")
        print("This is the optimal cutoff: ", threshold)

        # # Plot of a ROC curve
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", label="ROC curve (area = %f)" % roc_auc)
        plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic Curve")
        plt.legend(loc="lower right")
        plt.show()
        return ax.figure.savefig("boxplot_logodds.png")
