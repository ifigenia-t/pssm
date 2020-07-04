import argparse

from utils import compare_files

pep_window = 4

parser = argparse.ArgumentParser(description="")
parser.add_argument(
"--base_file", "-bf", help="base file to be used for the comparison", required=True
)
parser.add_argument(
    "--second_file", "-sf", help="file to be used for the comparison", required=True
)
args = parser.parse_args()


base_file = args.base_file
second_file = args.second_file


equality, f1_sd, f2_sd, sdf, sdfs = compare_files(base_file, second_file, pep_window)

print("Positions with significant SD for file: {} are: {}".format(base_file, f1_sd))
print("Positions with significant SD for file: {} are: {}".format(second_file, f2_sd))
print("Dataframes equal: {} ".format(equality))
print("Sum of square distance: {}".format(sdf))

for k, v in sdfs:
    #  if v <= 400:
    print("{} ===> {}".format(k, v))
