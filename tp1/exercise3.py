from fileHandling import read_csv, print_entire_df
import pandas as pd
from constants import Ex3_Headers, Ex3_Ranks

def run_exercise_3(file):
    df = read_csv(file)
    df = discretize_grades(df)
    compute_rank_probability(df)
    print_entire_df(compute_laplace_frequencies(df))

def discretize_grades(df):
    df[Ex3_Headers.GRE.value] = (df[Ex3_Headers.GRE.value] >= 500).astype(int)
    df[Ex3_Headers.GPA.value] = (df[Ex3_Headers.GPA.value] >= 3).astype(int)
    return df

def compute_rank_probability(df):
    return df[Ex3_Headers.RANK.value].value_counts(normalize=True,).rename_axis(Ex3_Headers.RANK.value).to_frame('probability')

# Probability of a grade given a rank P(Grade|Rank)
def compute_laplace_frequencies(df):
    # Create empty df
    _df = pd.DataFrame()
    for rank in Ex3_Ranks:
        # Get df filter for given rank
        filteredDf = df[df[Ex3_Headers.RANK.value] == rank.value][[Ex3_Headers.GRE.value, Ex3_Headers.GPA.value]]
        # Compute sum across columns
        laplaceDf = pd.DataFrame(data=filteredDf.sum(axis = 0, numeric_only = True)).transpose()
        # Apply Laplace to the frequencies given the current class
        laplaceDf = (laplaceDf + 1) / (filteredDf.shape[0] + len(Ex3_Ranks))
        # Append the rank
        laplaceDf[Ex3_Headers.RANK.value] = rank.value
        # Concat the new class probabilities
        _df = pd.concat([_df, laplaceDf], ignore_index = True)
    return _df