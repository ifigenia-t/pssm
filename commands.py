import json

from tqdm.auto import tqdm

from data_output import MultiComparison, output_data
from data_prep import (
    get_dfs_backwards_sliding,
    get_dfs_windowed,
    normalise_matrix,
    prepare_data,
    prepare_matrix,
)
from data_proc import process_data
from result import Result


def compare_combined_file(file1, similarity_metric, multi_metrics=False, correct_results_file=""):
    """
    Opens one json file containing multiple PSSMs and compares them all to eachother.
    """
    with open(file1) as json_file_1:
        data1 = json.load(json_file_1)

        multi_comp = MultiComparison(data1, data1)

        for base_pssm in tqdm(data1):
            base_file = json.dumps(data1[base_pssm]["pssm"])
        
            for pssm in tqdm(data1):
                try:
                    # for dms
                    json_pssm = json.dumps(
                        data1[pssm]["pssm"]
                    )

                    # for propPD
                    # json_pssm = json.dumps(
                    #     data1[pssm]["other_scoring_methods"]["log odds"]
                    # )

                    df1 = prepare_matrix(base_file)
                    df1_norm = normalise_matrix(df1)

                    df2 = prepare_matrix(json_pssm)
                    df2_norm = normalise_matrix(df2)

                    iters = get_dfs_backwards_sliding(df1_norm, df2_norm)

                    print("====> base:", base_pssm, "pssm:", pssm)
                    res = process_data(
                        iters, similarity_metric, multi_metrics=multi_metrics, 
                        correct_results_file=correct_results_file
                    )

                    # for dms
                    res.base_name = base_pssm
                    res.elm = pssm

                    res.quality = data1[pssm]["quality"]
                    res.consensus = data1[pssm]["consensus"]

                    # for propPD
                    # res.base_name = data1[base_pssm]["motif"]
                    # res.elm = data1[pssm]["motif"]
                    # res.quality = data1[pssm]["quality"]
                    # res.consensus = data1[pssm]["consensus"]

                    multi_comp.add(res)

                    if multi_metrics:
                        multi_comp.add_multi_metrics(res)

                    output_data(res.comparison_results)

                except Exception as e:
                    print(e)
    if multi_metrics:
        multi_comp.plot_multi_best_match()
        multi_comp.plot_multi_roc()
        multi_comp.plot_rank_boxplots()
    multi_comp.create_file(correct_results_file = "")

    # u_statistic, p_value = multi_comp.mann_whitney_u_test()
    # print("\nThis is the u statistic ", u_statistic)
    # print("\nThis is the p-value ", p_value)
    # if len(multi_comp.match) != 0 or len(multi_comp.mismatch) != 0:
    #     multi_comp.plot_match()
    #     multi_comp.plot_ROC()


def compare_single_to_combined_file(file1, file2, similarity_metric):
    """
    Opens two files, one containing one single PSSM and the other multiple PSSMs.
    Compares the single PSSM to all the multiple ones a
    nd returns the best comparison result.
    """
    with open(file1) as json_file_1, open(file2) as json_file_2:
        data1 = json.load(json_file_1)
        data2 = json.load(json_file_2)

    base_file = file1
    x = base_file.split("/")
    x = x[1].split(".")
    filename = x[0]

    comp = MultiComparison(data1, use_index=False, file_name=filename)
    print("Base file ", base_file)

    for pssm in tqdm(data2):
        try:
            json_pssm = json.dumps(data2[pssm]["other_scoring_methods"]["log odds"])

            print("-----> ", data2[pssm]["motif"])

            df1 = prepare_matrix(base_file)
            df1_norm = normalise_matrix(df1)

            df2 = prepare_matrix(json_pssm)
            df2_norm = normalise_matrix(df2)

            iters = get_dfs_backwards_sliding(df1_norm, df2_norm)

            res = process_data(iters, similarity_metric)

            res.base_name = filename
            res.elm = data2[pssm]["motif"]
            res.quality = data2[pssm]["quality"]
            res.consensus = data2[pssm]["consensus"]

            comp.add(res)

            output_data(res.comparison_results)

        except Exception as e:
            print("this is what failed")
            print(e)

    comp.create_file()


def compare_two_combined_new(
    file1, file2, similarity_metric, correct_results_file="", multi_metrics=False
):
    """
    Opens two json files containing multiple PSSMs and compares them all to eachother.
    Output is a csv table containing the best results of the comparison.
    """
    with open(file1) as json_file_1, open(file2) as json_file_2:
        data1 = json.load(json_file_1)
        data2 = json.load(json_file_2)

        multi_comp = MultiComparison(
            data1, data2, correct_results_file=correct_results_file
        )
        print("correct_result_file ", correct_results_file)
        for base_pssm in tqdm(data1):
            base_file = json.dumps(data1[base_pssm]["pssm"])
            print("-----> ", base_pssm)
            # print("-----> ", data1[base_pssm]["motif"])

            for pssm in tqdm(data2):
                try:
                    json_pssm = json.dumps(
                        data2[pssm]["other_scoring_methods"]["log odds"]
                    )

                    print("-----> ", data2[pssm]["motif"])

                    df1 = prepare_matrix(base_file)
                    df1_norm = normalise_matrix(df1)

                    df2 = prepare_matrix(json_pssm)
                    df2_norm = normalise_matrix(df2)

                    iters = get_dfs_backwards_sliding(df1_norm, df2_norm)

                    res = process_data(
                        iters, similarity_metric, multi_metrics=multi_metrics
                    )

                    # res.base_name = data1[base_pssm]["motif"]
                    res.base_name = base_pssm
                    res.elm = data2[pssm]["motif"]
                    res.quality = data2[pssm]["quality"]
                    res.consensus = data2[pssm]["consensus"]

                    multi_comp.add(res)
                    if multi_metrics:
                        multi_comp.add_multi_metrics_file(res)

                    output_data(res.comparison_results)

                except Exception as e:
                    print("this is what failed 2")
                    print(e)

    if multi_metrics:
        multi_comp.plot_multi_best_match_file()
        multi_comp.plot_multi_roc()
        multi_comp.plot_rank_boxplots()
    multi_comp.create_file(correct_results_file)

    # u_statistic, p_value = multi_comp.mann_whitney_u_test()
    # print("\nThis is the u statistic ", u_statistic)
    # print("\nThis is the p-value ", p_value)
    # if len(multi_comp.match) != 0 or len(multi_comp.mismatch) != 0:
    #     multi_comp.plot_match()
    #     multi_comp.plot_ROC()


def compare_two_files_new(
    base_file, second_file, similarity_metric, pep_window, backwards_reading=False
):
    """
    Calculate all the comparisons for two PSSMs
    """
    df1 = prepare_matrix(base_file)
    df1_norm = normalise_matrix(df1)

    df2 = prepare_matrix(second_file)
    df2_norm = normalise_matrix(df2)

    iters = get_dfs_backwards_sliding(df1_norm, df2_norm)

    res = process_data(iters, similarity_metric)

    output_data(res.comparison_results)
