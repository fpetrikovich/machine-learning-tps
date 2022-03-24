from fileHandling import read_csv, print_entire_df
import pandas as pd
from constants import Ex3_Headers, Ex3_Ranks

def run_exercise_3(file):
    df = read_csv(file)
    df = discretize_grades(df)
    print_entire_df(compute_rank_probability(df))
    print_entire_df(compute_grade_frequencies(df))
    print_entire_df(compute_admision_frequencies(df))

def discretize_grades(df):
    df[Ex3_Headers.GRE.value] = (df[Ex3_Headers.GRE.value] >= 500).astype(int)
    df[Ex3_Headers.GPA.value] = (df[Ex3_Headers.GPA.value] >= 3).astype(int)
    return df

def compute_rank_probability(df):
    return df[Ex3_Headers.RANK.value].value_counts(normalize=True,).rename_axis(Ex3_Headers.RANK.value).to_frame('probability')

# Probability of a grade given a rank P(Grade|Rank)
def compute_grade_frequencies(df):
    posible_grade_classes = [0, 1]
    # Create empty df
    _df = pd.DataFrame()
    for rank in Ex3_Ranks:
        # Get df filter for given rank
        filtered_df = df[df[Ex3_Headers.RANK.value] == rank.value][[Ex3_Headers.GRE.value, Ex3_Headers.GPA.value]]
        # Compute sum across columns
        laplace_df = pd.DataFrame(data=filtered_df.sum(axis = 0, numeric_only = True)).transpose()
        # Apply Laplace to the frequencies given the current class
        laplace_df = (laplace_df + 1) / (filtered_df.shape[0] + len(posible_grade_classes))
        # Append the negated probabilities
        laplace_df['low_'+Ex3_Headers.GRE.value] = laplace_df.apply(lambda laplace_df: compute_negation_probability(laplace_df, Ex3_Headers.GRE), axis=1)
        laplace_df['low_'+Ex3_Headers.GPA.value] = laplace_df.apply(lambda laplace_df: compute_negation_probability(laplace_df, Ex3_Headers.GPA), axis=1)
        # Append the rank
        laplace_df[Ex3_Headers.RANK.value] = rank.value
        # Concat the new class probabilities
        _df = pd.concat([_df, laplace_df], ignore_index = True)
    return _df

# Probability of a admision given a rank, gpa, and gre
def compute_admision_frequencies(df):
    posible_admit_classes = [0, 1]
    grade_combinations = [[0, 0], [0, 1], [1, 0], [1,1]]
    # Create empty df
    _df = pd.DataFrame()
    for rank in Ex3_Ranks:
        for grades in grade_combinations:
            # Get df filter for given rank
            filtered_df = df[(df[Ex3_Headers.RANK.value] == rank.value) & (df[Ex3_Headers.GRE.value] == grades[0]) & (df[Ex3_Headers.GPA.value] == grades[1])]
            # Take only the column we want to analyze
            filtered_df = filtered_df[[Ex3_Headers.ADMIT.value]]
            # Compute sum across columns
            laplace_df = pd.DataFrame(data=filtered_df.sum(axis = 0)).transpose()
            # Apply Laplace to the frequencies given the current class
            laplace_df = (laplace_df + 1) / (filtered_df.shape[0] + len(posible_admit_classes))
            # Append the probability of no admition
            laplace_df['no_'+Ex3_Headers.ADMIT.value] = laplace_df.apply(lambda laplace_df: compute_negation_probability(laplace_df, Ex3_Headers.ADMIT), axis=1)
            # Append the fixed values
            laplace_df[Ex3_Headers.RANK.value] = rank.value
            laplace_df[Ex3_Headers.GRE.value] = grades[0]
            laplace_df[Ex3_Headers.GPA.value] = grades[1]
            # Concat the new class probabilities
            _df = pd.concat([_df, laplace_df], ignore_index = True)
    return _df

def compute_negation_probability(df, column_header):
    return 1 - df[column_header.value]