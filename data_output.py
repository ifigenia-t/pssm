import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


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
        self.best_df.to_csv("propd-vs-elm-best.csv")
        if hasattr(self, "all_df"):
            self.all_df.to_csv("propd-vs-elm-all.csv")

    def plot_match(self):
        match_df = pd.DataFrame(self.match, columns=["Match"])
        mismatch_df = pd.DataFrame(self.mismatch, columns=["Mismatch"])

        comb_df = pd.concat([match_df, mismatch_df], axis=1)
        # print(comb_df)
        mdf = pd.melt(comb_df)
        # print(mdf)
        mdf = mdf.rename(columns={"variable": "Identification", "value": "Comparison"})
        print(mdf)
        ax = sns.boxplot(x="Identification", y="Comparison", data=mdf)

        plt.show()

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
                self.match.append(result.comparison_results[1])
            else:
                self.mismatch.append(result.comparison_results[1])
                print(result.base_name, result.elm, "miss")

        if hasattr(self, "all_df"):
            print("adding 2...", result.elm)
            self.all_df.at[
                result.elm, result.base_name
            ] = result.comparison_results[1]

