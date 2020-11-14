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


def compare_combined_file(file1):
    pass


def compare_single_to_combined_file(file1, file2):
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

            res = process_data(iters)
            
            res.base_name = filename
            res.elm =  data2[pssm]["motif"]
            res.quality = data2[pssm]["quality"]
            res.consensus = data2[pssm]["consensus"]

            comp.add(res)

            output_data(res.comparison_results)

        except Exception as e:
            print(e)

    comp.create_file()


def compare_two_combined_new(file1, file2, correct_results_file=""):
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

        for base_pssm in tqdm(data1):
            base_file = json.dumps(data1[base_pssm]["pssm"])
            print("-----> ", data1[base_pssm]["motif"])

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

                    res = process_data(iters)

                    res.base_name = data1[base_pssm]["motif"]
                    res.elm =  data2[pssm]["motif"]
                    res.quality = data2[pssm]["quality"]
                    res.consensus = data2[pssm]["consensus"]

                    multi_comp.add(res)

                    output_data(res.comparison_results)

                except Exception as e:
                    print(e)

    multi_comp.create_file()
    if len(multi_comp.match) != 0 or len(multi_comp.mismatch) != 0:
        multi_comp.plot_match()


def compare_two_files_new(base_file, second_file, pep_window, backwards_reading=False):
    """
    Calculate all the comparisons for two PSSMs
    """
    df1 = prepare_matrix(base_file)
    df1_norm = normalise_matrix(df1)

    df2 = prepare_matrix(second_file)
    df2_norm = normalise_matrix(df2)

    iters = get_dfs_backwards_sliding(df1_norm, df2_norm)

    res = process_data(iters)

    output_data(res.comparison_results)
