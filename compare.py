import argparse

from utils import compare_combined_file, compare_two_files

pep_window = 4

parser = argparse.ArgumentParser(description="")
parser.add_argument(
    "--base_file", "-bf", help="base file to be used for the comparison", required=True
)
parser.add_argument(
    "--second_file", "-sf", help="file to be used for the comparison",
)
parser.add_argument(
    "--combined_file",
    "-cf",
    help="file that contains multile json objects to be used for the comparison",
)
args = parser.parse_args()

if (not args.second_file) and (not args.combined_file):
    parser.error("--second_file or --combined_file need to be defined")

base_file = args.base_file

if args.second_file:
    second_file = args.second_files
    equality, f1_sd, f2_sd, ssd, sdfs = compare_two_files(
        base_file, second_file, pep_window
    )
    print("Positions with significant SD for file: {} are: {}".format(base_file, f1_sd))
    print(
        "Positions with significant SD for file: {} are: {}".format(second_file, f2_sd)
    )
    print("Dataframes equal: {} ".format(equality))
    print("Sum of square distance: {}".format(ssd))
    for k, v in sdfs:
        print("{} ===> {}".format(k, v))


if args.combined_file:
    combined_file = args.combined_file
    results = compare_combined_file(base_file, combined_file, pep_window)

    res_by_sdf = results[0]
    results.sort(key=lambda x: x["ssd"])
    res_by_ssd = results[0]

    print(
        "---> Window Calculations = Base: {} Second: {} SSD: {} SDF: {}".format(
            res_by_sdf["base"],
            res_by_sdf["second"],
            res_by_sdf["ssd"],
            res_by_sdf["sdf"],
        )
    )
    print(
        "---> Whole Matrix Comparisons = Base: {} Second: {} SSD: {} SDF: {}".format(
            res_by_ssd["base"],
            res_by_ssd["second"],
            res_by_ssd["ssd"],
            res_by_ssd["sdf"],
        )
    )
