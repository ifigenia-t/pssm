import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.indexes import base
from scipy.stats.mstats_basic import pearsonr
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_auc_score, auc, roc_curve


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

        self.best_df = pd.DataFrame(columns=col_names, index=data_1_names)
        if data_file_2 is not "":
            data_2_names = []
            for pssm in data_file_2:
                data_2_names.append(data_file_2[pssm]["motif"])
            self.all_df = pd.DataFrame(columns=data_1_names, index=data_2_names)

        if correct_results_file is not "":
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
            self.all_df.at[result.elm, result.base_name] = result.comparison_results[1]

    def add_multi_metrics_file(self, result):
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

    def add_multi_metrics(self, result):
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

    def plot_multi_roc(self):
        print(json.dumps(self.multi_metrics))

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
