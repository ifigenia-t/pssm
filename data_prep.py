import numpy as np
import pandas as pd


def prepare_data():
    pass


# Dataframe preparation
def prepare_matrix(filename):
    """
    Opens a json file and converts it to a dataframe of the correct format
    """
    df = pd.read_json(filename)
    df = df.dropna(axis=1, how="all")
    df_proc = df.T
    df_proc = df_proc.dropna()
    df_proc = df_proc.sort_index(axis=0)
    return df_proc


def normalise_matrix(df):
    """
    Performs normalisation calculations on each column of the matrix and returns
    the normalised matrix
    Pandas automatically applies colomn-wise functions.
    """
    normalized_df = df - df.min()
    normalized_df = normalized_df / normalized_df.sum()
    return normalized_df


def convert_to_df(data, tag):
    """
    Coverts data into dataframes with a given tag as an index name.
    """
    df = pd.DataFrame(data)
    df = df.T
    df = df.rename(index={0: tag})
    return df



def get_dfs_backwards_sliding(df1, df2):
    big_df = df1
    small_df = df2

    iters = []

    if len(df2.columns) > len(df1.columns):
        big_df = df2
        small_df = df1

    
    for length in range(0, len(small_df.columns)):
        big_x = 0
        big_y = length
        small_x = len(small_df.columns)-length-1
        small_y = len(small_df.columns)-1

        a = big_df[big_x:big_y]
        b = small_df[small_x:small_y]

        a = big_df.loc[:, big_x:big_y]
        b = small_df.loc[:, small_x: small_y]

        key = "{}:{} - {}:{}".format(big_x, big_y, small_x, small_y)

        a.columns = [ind for ind in range(big_x, big_y + 1)]
        b.columns = [ind for ind in range(small_x, small_y + 1)]

        iters.append((key, a, b))

    for i in range(1, len(big_df.columns)-len(small_df.columns)+1):
        big_x = i
        big_y =  len(small_df.columns) -1  + i

        a = big_df[big_x:big_y]
        b = small_df

        a = big_df.loc[:, big_x:big_y]
        b = small_df.loc[:, small_x: small_y]

        key = "{}:{} - {}:{}".format(big_x, big_y, small_x, small_y)

        a.columns = [ind for ind in range(big_x, big_y + 1)]
        b.columns = [ind for ind in range(small_x, small_y + 1)]

        iters.append((key, a, b))

    for i in reversed(range(0, len(small_df.columns)-1)):
        big_x = len(big_df.columns)-i-1
        big_y = len(big_df.columns)-1

        small_x = 0
        small_y = i
        
        a = big_df[big_x:big_y]
        b = small_df[small_x:small_y]

        a = big_df.loc[:, big_x:big_y]
        b = small_df.loc[:, small_x: small_y]

        key = "{}:{} - {}:{}".format(big_x, big_y, small_x, small_y)

        a.columns = [ind for ind in range(big_x, big_y + 1)]
        b.columns = [ind for ind in range(small_x, small_y + 1)]

        iters.append((key, a, b))

    return iters


def get_dfs_windowed(df1, df2, pep_window):
    it_window_a = len(df1.columns) - pep_window
    it_window_b = len(df2.columns) - pep_window

    iters = []

    for i in range(0, it_window_a + 1):
        for j in range(0, it_window_b + 1):
            key = "{}:{} - {}:{}".format(i, i + pep_window - 1, j, j + pep_window - 1)
            a, b = get_df_window(df1, df2, pep_window, i, j)
            iters.append((key, a, b))

    return iters


def get_df_window(df1, df2, pep_window, i, j):
    """
    Slices the dataframes according to a given window size.
    """
    a = df1.loc[:, i : i + pep_window - 1]
    b = df2.loc[:, j : j + pep_window - 1]

    a.columns = [ind for ind in range(0, len(a.columns))]
    b.columns = [ind for ind in range(0, len(b.columns))]

    return a, b

def gini_weight(df):
    """
    Calculates the Gini Coefficient which is a measure of statistical dispersion.
    Gini = 1 means maximal inequallity
    Gini = 0 means perfect equallity where all the values are the same. 
    """
    # gini_df = pd.DataFrame(columns=df.columns)
    d = {}
    for col in df.columns:
        col_list = df[col].to_numpy()
        mad = np.abs(np.subtract.outer(col_list, col_list)).mean()
        rmad = mad / col_list.mean()
        g1 = 0.5 * rmad
        d[col] = g1
        # TODO must fix, takes time

    d_df = pd.DataFrame.from_dict(d, orient="index")
    # d_df = d_df.round(decimals=2)

    return d_df


