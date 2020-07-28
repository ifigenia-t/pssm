import argparse

from utils import compare_combined_file, compare_two_files, print_df_ranges

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
    second_file = args.second_file
    (
        equality,
        f1_sd,
        f2_sd,
        ssd,
        sdfs,
        df1,
        df2,
        ssd_print,
        pearsons,
        spearmans
    ) = compare_two_files(base_file, second_file, pep_window)
    print("Positions with significant SD for file: {} are: {}".format(base_file, f1_sd))
    print(
        "Positions with significant SD for file: {} are: {}".format(second_file, f2_sd)
    )
    print("Dataframes equal: {} ".format(equality))
    print("Sum of square distance: {}".format(ssd))
    res_by_sdf = sdfs[0]
    regions = res_by_sdf[0]
    region_a, region_b = regions.split(" - ")
    print("{} ===> {}".format(res_by_sdf[0], res_by_sdf[1]))
    print_df_ranges(df1, region_a, ssd_print, pearsons, spearmans)
    print_df_ranges(df2, region_b, ssd_print, pearsons, spearmans)

if args.combined_file:
    combined_file = args.combined_file
    results = compare_combined_file(base_file, combined_file, pep_window)

    res_by_sdf = results[0]
    results.sort(key=lambda x: x["ssd"])
    res_by_ssd = results[0]
    regions = res_by_sdf["sdf"][0]
    region_a, region_b = regions.split(" - ")

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
    print_df_ranges(
        res_by_sdf["df1"],
        region_a,
        res_by_ssd["ssd_print"],
        res_by_ssd["pearsons"],
        res_by_ssd["spearmans"],
    )
    print_df_ranges(
        res_by_sdf["df2"],
        region_b,
        res_by_ssd["ssd_print"],
        res_by_ssd["pearsons"],
        res_by_ssd["spearmans"],
    )

