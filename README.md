# CompariPSSM
Matrix Comparison 

CompariPSSM is a tool that calculates the similarity between two Position Specific Scoring Matrices (PSSMs).


## Getting Started

```
git clone https://github.com/ifigenia-t/pssm.git
cd pssm 
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Workflow
The default input of CompariPSSM is two PSSMs. The PSSMs are normalised so that the sum of all the values in each position (the sum of every column) equals one. Windows for PSSM comparison are defined using the PSSM-PSSM comparison window definition as the sliding window. 
For each window, a comparison score is calculated. The comparison score for a single position is calculated as the product of the similarity and the two Gini coefficients. The comparison score of the window is calculated as the sum of the positional scores in the PSSM windows. The highest scoring alignment is considered the optimal alignment. 

## Execution

All by all comparison of two files containing multiple PSSMs: 
```
python compare.py -sm pearson -two sample_data/small_pssm_set.json -two sample_data/big_pssm_set.json
```

Pairwise comparison of two files containing 1 PSSM each:
```
python compare.py -sm pearson -bf sample_data/first_pssm.json -sf sample_data/second_pssm.json
```

All by all comparison of one file containing multiple PSSMs:
```
python compare.py -sm pearson -sif sample_data/big_pssm_set.json
```

## CLI Arguments
```
python compare.py --help                                                                                     
usage: compare.py [-h] [--base_file BASE_FILE] [--second_file SECOND_FILE] [--combined_file COMBINED_FILE] [--single_file SINGLE_FILE] [--two_comb_files TWO_COMB_FILES] [--peptide_window PEPTIDE_WINDOW] [--boxplot BOXPLOT]
                  [--correct_results_file CORRECT_RESULTS_FILE] [--multi_metrics MULTI_METRICS] [--similarity_metric SIMILARITY_METRIC]

optional arguments:
  -h, --help            show this help message and exit
  --base_file BASE_FILE, -bf BASE_FILE
                        base file to be used for the comparison
  --second_file SECOND_FILE, -sf SECOND_FILE
                        file to be used for the comparison
  --combined_file COMBINED_FILE, -cf COMBINED_FILE
                        file that contains multile json objects to be used for the comparison
  --single_file SINGLE_FILE, -sif SINGLE_FILE
                        file that contains multile json objects to be used for the comparison with each other
  --two_comb_files TWO_COMB_FILES, -two TWO_COMB_FILES
                        Two files that contain multiple json objects to be used for comparison with each other
  --peptide_window PEPTIDE_WINDOW, -pw PEPTIDE_WINDOW
                        The length of the window of the PSSM comparison
  --boxplot BOXPLOT, -box BOXPLOT
                        Boxplot the important vs the unimportant positions
  --correct_results_file CORRECT_RESULTS_FILE, -crf CORRECT_RESULTS_FILE
                        Correct results file to compaire against
  --multi_metrics MULTI_METRICS, -mm MULTI_METRICS
                        Return multiple similarity metrics from comparison
  --similarity_metric SIMILARITY_METRIC, -sm SIMILARITY_METRIC
                        Similarity metrics used for the comparison
```

## Author
Ifigenia Tsitsa
