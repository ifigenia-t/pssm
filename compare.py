import argparse
import operator

from utils import (calc_brute_force_window, calc_gini_windows,
                   compare_combined_file, compare_single_file,
                   compare_two_files, normalise_matrix, prepare_matrix,
                   print_df_ranges)

# pep_window = 4
buffer = 1

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--base_file", "-bf", help="base file to be used for the comparison",
)
parser.add_argument(
    "--second_file", "-sf", help="file to be used for the comparison",
)
parser.add_argument(
    "--combined_file",
    "-cf",
    help="file that contains multile json objects to be used for the comparison",
)
parser.add_argument(
    "--single_file",
    "-sif",
    help="file that contains multile json objects to be used for the comparison with each other",
)
parser.add_argument("--peptide_window", "-pw", help="The length of the window of the PSSM comparison")
args = parser.parse_args()

if args.peptide_window:
    pep_window = args.peptide_window
 


if args.base_file:
    if (not args.second_file) and (not args.combined_file):
        parser.error("--second_file or --combined_file need to be defined")

    base_file = args.base_file

elif args.second_file or args.combined_file:
    parser.error("--base_file needs to be defined")

if args.second_file:
    second_file = args.second_file
    df2 = prepare_matrix(second_file)
    df1 = prepare_matrix(base_file)

    df1 = normalise_matrix(df1)
    df2 = normalise_matrix(df2)

    pep_windows = []

    try:
        pep_window
    except:
        pep_windows = calc_gini_windows(df1,df2)
        print("Pep_windows: ", pep_windows)
    else:
        pep_windows.append(int(pep_window))

    optimal_window = {}
    for window in pep_windows:
        print("---> calculating for window: {} ".format(window))
        (
            equality,
            f1_sd,
            f2_sd,
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
        ) = compare_two_files(base_file, second_file, window)
        print("Positions with significant SD for file: {} are: {}".format(base_file, f1_sd))
        print(
            "Positions with significant SD for file: {} are: {}".format(second_file, f2_sd)
        )
        print("Dataframes equal: {} ".format(equality))
        print("Sum of square distance: {}".format(ssd_global))

        res_best = comparison_results[0]
        regions = res_best[0]
        region_a, region_b = regions.split(" - ")

        optimal_window[window] = res_best[1]    

        print("{} ===> {}".format(res_best[0], res_best[1]))
        print_df_ranges(
            df1, region_a, ssd, pearsons, spearmans, kendalls, dot_products, kl_divergence
        )
        print_df_ranges(
            df2, region_b, ssd, pearsons, spearmans, kendalls, dot_products, kl_divergence
        )
    norm_opt_window = {k: v / k for k, v in optimal_window.items()}
    opt_window_ordered = sorted(norm_opt_window.items(), key=lambda x: x[1], reverse=True)
    print(optimal_window)
    print("Optimal window normalised: ", norm_opt_window)
    print("Best window ", opt_window_ordered[0], "\n")


if args.combined_file:
    combined_file = args.combined_file
    results = compare_combined_file(base_file, combined_file, pep_window)

    res_best = results[0]

    # results.sort(key=lambda x: x["ssd"])
    # res_by_ssd = results[0]

    regions = res_best["comparison_results"][0]
    region_a, region_b = regions.split(" - ")

    # print(
    #     "---> Whole Matrix Comparisons = Base: {} Second: {} SSD: {} SDF: {}".format(
    #         res_by_ssd["base"],
    #         res_by_ssd["second"],
    #         res_by_ssd["ssd_global"],
    #         res_by_ssd["sdf"],
    #     )
    # )

    # for result in results:
    #     print("Comparison Score = {}, ELM motif = {}".format(result["comparison_results"], result["second"]))

    print(
        "---> Window Calculations = Base: {} Second: {} SSD: {} Comparison: {}".format(
            res_best["base"],
            res_best["second"],
            res_best["ssd_global"],
            res_best["comparison_results"],
        )
    )
    print_df_ranges(
        res_best["df1"],
        region_a,
        res_best["ssd"],
        res_best["pearsons"],
        res_best["spearmans"],
        res_best["kendalls"],
        res_best["dot_products"],
        res_best["kl_divergence"],
    )
    print_df_ranges(
        res_best["df2"],
        region_b,
        res_best["ssd"],
        res_best["pearsons"],
        res_best["spearmans"],
        res_best["kendalls"],
        res_best["dot_products"],
        res_best["kl_divergence"],
    )

if args.single_file:
    single_file = args.single_file
    results = compare_single_file(single_file, pep_window)

    for result in results:
         print("1st PSSM = {}, 2nd PSSM = {}, Comparison Score = {}".format(result["base"], result["second"], result["comparison_results"]))
