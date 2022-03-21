from fileHandling import read_data
from constants import Ex1_Headers, Ex1_Nacionalidad
import pandas as pd
import numpy as np

def compute_laplace_frequencies(df):
    _df = pd.DataFrame()
    for nacionality in Ex1_Nacionalidad:
        filteredDf = df[df[Ex1_Headers.NACIONALIDAD.value] == nacionality.value]
        tmp = pd.DataFrame(data=filteredDf.sum(axis = 0, numeric_only = True)).transpose()
        tmp = (tmp + 1) / (filteredDf.shape[0] + len(Ex1_Nacionalidad))
        tmp[Ex1_Headers.NACIONALIDAD.value] = nacionality.value
        _df = pd.concat([_df, tmp], ignore_index = True)
    return _df

def run_exercise_1(file):
    # Get the data
    df = read_data(file)
    print(compute_laplace_frequencies(df))