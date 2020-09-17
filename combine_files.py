import json
import os
from pathlib import Path

data = {}

data_path = "data/ProPPD_PSSMs/"

pathlist = Path(data_path).glob("**/**/*.json")
for path in pathlist:
    # because path is object not string
    path_in_str = str(path)
    filename = os.path.basename(path_in_str)
    elems = filename.split(".")[0].split("_")
    pssm = "_".join(elems[0:2])
    with open(path_in_str, "r") as pssm_file:
        pssm_data = json.load(pssm_file)
        pssm_data_out = {
            "consensus": pssm_data["data"]["motif"],
            "motif": pssm,
            "pssm": pssm_data["data"]["pssm"],
        }

        print(pssm)
        data[pssm] = pssm_data_out

with open("data.json", "w") as outfile:
    json.dump(data, outfile)
