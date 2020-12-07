import argparse
import operator

from commands import compare_two_combined_new, compare_two_files_new, compare_single_to_combined_file
from utils import (calc_gini_windows, compare_combined_file,
                   compare_single_file, compare_two_combined,
                   compare_two_files, normalise_matrix,
                   plot_important_positions, prepare_matrix, print_df_ranges)

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

parser.add_argument(
    "--two_comb_files",
    "-two",
    help="Two files that contain multiple json objects to be used for comparison with each other",
    action="append",
)

parser.add_argument(
    "--peptide_window", "-pw", help="The length of the window of the PSSM comparison"
)
parser.add_argument(
    "--boxplot", "-box", help="Boxplot the important vs the unimportant positions"
)
parser.add_argument(
    "--correct_results_file", "-crf", help="Correct results file to compaire against"
)
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

    compare_two_files_new(base_file, second_file, 4)


if args.combined_file:
    combined_file = args.combined_file
    compare_single_to_combined_file(base_file, combined_file)

if args.two_comb_files:
    if len(args.two_comb_files) != 2:
        parser.error("--two_comb_files needs two values")

    correct_results_file = ""
    if args.correct_results_file:
        correct_results_file = args.correct_results_file

    print(args.two_comb_files)
    file1, file2 = args.two_comb_files[0], args.two_comb_files[1]
    compare_two_combined_new(file1, file2, correct_results_file)

if args.single_file:
    single_file = args.single_file
    results = compare_single_file(single_file, pep_window=0)

    for result in results:
        print(
            "1st PSSM = {}, 2nd PSSM = {}, Comparison Score = {}".format(
                result["base"], result["second"], result["comparison_results"]
            )
        )


if args.boxplot:
    boxplot = args.boxplot
    plot_important_positions(boxplot)
