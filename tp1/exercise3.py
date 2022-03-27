from fileHandling import read_csv, print_entire_df
import pandas as pd
from constants import Ex3_Headers, Ex3_Ranks, Ex3_Negated_Headers

def run_exercise_3(file):
    df = read_csv(file)
    df = discretize_grades(df)
    p_rank_table = compute_rank_probability(df)
    p_grade_table = compute_grade_frequencies(df)
    p_admit_table = compute_admision_frequencies(df)
    print_entire_df(p_admit_table)
    # Item A
    print("~~~~~~~~~~~~~~~~~~~~ ITEM A ~~~~~~~~~~~~~~~~~~~~\n")
    probability_no_admision_given_rank(p_grade_table, p_admit_table, Ex3_Ranks.FIRST.value)
    probability_no_admision_given_rank(p_grade_table, p_admit_table, Ex3_Ranks.SECOND.value)
    probability_no_admision_given_rank(p_grade_table, p_admit_table, Ex3_Ranks.THIRD.value)
    probability_no_admision_given_rank(p_grade_table, p_admit_table, Ex3_Ranks.FOURTH.value)
    # Item B
    print("~~~~~~~~~~~~~~~~~~~~ ITEM B ~~~~~~~~~~~~~~~~~~~~\n")
    probability_admision_given_rank_and_grade(p_admit_table, Ex3_Ranks.FIRST.value, 600, 4)
    probability_admision_given_rank_and_grade(p_admit_table, Ex3_Ranks.FIRST.value, 400, 4)
    probability_admision_given_rank_and_grade(p_admit_table, Ex3_Ranks.FIRST.value, 600, 2.5)
    probability_admision_given_rank_and_grade(p_admit_table, Ex3_Ranks.FIRST.value, 400, 2.5)
    probability_admision_given_rank_and_grade(p_admit_table, Ex3_Ranks.SECOND.value, 450, 3.5)
    probability_admision_given_rank_and_grade(p_admit_table, Ex3_Ranks.FOURTH.value, 400, 2.8)

def discretize_grades(df):
    df[Ex3_Headers.GRE.value] = (df[Ex3_Headers.GRE.value] >= 500).astype(int)
    df[Ex3_Headers.GPA.value] = (df[Ex3_Headers.GPA.value] >= 3).astype(int)
    return df

def compute_rank_probability(df):
    return df[Ex3_Headers.RANK.value].value_counts(normalize=True,).rename_axis(Ex3_Headers.RANK.value).reset_index(name='probability')

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
        laplace_df[Ex3_Negated_Headers.GRE.value] = laplace_df.apply(lambda laplace_df: compute_negation_probability(laplace_df, Ex3_Headers.GRE), axis=1)
        laplace_df[Ex3_Negated_Headers.GPA.value] = laplace_df.apply(lambda laplace_df: compute_negation_probability(laplace_df, Ex3_Headers.GPA), axis=1)
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
            # Useful to see the amount of data is very different => leads to some probabilities that do not make sense
            print("Data per category: " + str(rank), str(grades), str(filtered_df.shape[0]))
            # Apply Laplace to the frequencies given the current class
            laplace_df = (laplace_df + 1) / (filtered_df.shape[0] + len(posible_admit_classes))
            # Append the probability of no admition
            laplace_df[Ex3_Negated_Headers.ADMIT.value] = laplace_df.apply(lambda laplace_df: compute_negation_probability(laplace_df, Ex3_Headers.ADMIT), axis=1)
            # Append the fixed values
            laplace_df[Ex3_Headers.RANK.value] = rank.value
            laplace_df[Ex3_Headers.GRE.value] = grades[0]
            laplace_df[Ex3_Headers.GPA.value] = grades[1]
            # Concat the new class probabilities
            _df = pd.concat([_df, laplace_df], ignore_index = True)
    return _df

def compute_negation_probability(df, column_header):
    return 1 - df[column_header.value]

# Calcular la probabilidad de que una persona que proviene de una escuela con rango
# 1 no haya sido admitida en la universidad.
# P(A=0|R=1) = P(A=0, R=1)/P(R=1) = sum{x=0,1 y=0,1}(P(A=0, R=1, GPA=x, GRE=y))/P(R=1)
# Primer termino: P(A=0|R=1 y GPA y GRE) * P(GPA y GRE | R=1) * P(R=1)
# Como GPA y GRE son independientes dado R => P(GPA y GRE | R=1) = P(GPA|R) * P(GRE|R)
def probability_no_admision_given_rank(grade_table, admit_table, rank_value):
    grade_combinations = [[0, 0], [0, 1], [1, 0], [1,1]]
    result = 0

    # Conditions
    rank_filtering = lambda df: rank_df_filtering(df, rank_value)

    # Filtered grade table by rank 1
    filtered_grade_df = rank_filtering(grade_table)
    # Find P(GPA | R = 1) and P(GRE | R = 1) for grades 0 and 1
    p_high_gre_given_r = filtered_grade_df[Ex3_Headers.GRE.value].values[0]
    p_high_gpa_given_r = filtered_grade_df[Ex3_Headers.GPA.value].values[0]
    p_low_gre_given_r = filtered_grade_df[Ex3_Negated_Headers.GRE.value].values[0]
    p_low_gpa_given_r = filtered_grade_df[Ex3_Negated_Headers.GPA.value].values[0]
    
    # Filter admit table by rank 1
    filtered_admit_df = rank_filtering(admit_table)
    # Find all the terms P(A=0 | R=1, GRE, GPA) and calculate term
    for (gre, gpa) in grade_combinations:
        no_admit_prob_row = grade_df_filtering(filtered_admit_df, gre, gpa)
        # Term = P(A=0|R=1,GRE,GPA) * P(GRE|R=1) * P(GPA|R=1)
        term = no_admit_prob_row[Ex3_Negated_Headers.ADMIT.value].values[0]
        # P(GRE|R=1)
        p_gre_given_r = p_high_gre_given_r if gre == 1 else p_low_gre_given_r
        # P(GPA|R=1)
        p_gpa_given_r = p_high_gpa_given_r if gpa == 1 else p_low_gpa_given_r
        # Calculate the term
        term = term * p_gre_given_r * p_gpa_given_r
        # Add all the terms
        result += term

    print("################ P(A=0 | R=" + str(rank_value) + ") ################")
    print("Probability of not being admitted given they went to a rank " + str(rank_value) + " school P(A=0 | R=1):")
    print(result)
    print()
    return result

def rank_df_filtering(df, value):
    return df[df[Ex3_Headers.RANK.value] == value]

def grade_df_filtering(df, gre, gpa):
    return df[(df[Ex3_Headers.GRE.value] == gre) & (df[Ex3_Headers.GPA.value] == gpa)]

def probability_admision_given_rank_and_grade(admit_table, rank, gre, gpa):
    discretized_gre = 1 if gre >= 500 else 0
    discretized_gpa = 1 if gpa >= 3 else 0

    filtered_df = grade_df_filtering(rank_df_filtering(admit_table, rank), discretized_gre, discretized_gpa)
    result = filtered_df[Ex3_Headers.ADMIT.value].values[0]

    print("################ P(A=1 | R=" + str(rank) + ", GRE=" + str(gre) + ", GPA=" + str(gpa) + ") ################")
    print("Probability of being admitted given the specified conditions:")
    print(result)
    print()
