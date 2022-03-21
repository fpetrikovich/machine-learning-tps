from fileHandling import read_data
from constants import Ex1_Headers, Ex1_Nacionalidad
import pandas as pd

def build_examples(columns):
    sample_1 = pd.DataFrame(data=[[1,0,1,1,0,'?']], columns=columns)
    sample_2 = pd.DataFrame(data=[[0,1,1,0,1,'?']], columns=columns)
    return sample_1, sample_2

def get_df_for_class(df, _class):
    return df[df[Ex1_Headers.NACIONALIDAD.value] == _class]

def compute_laplace_frequencies(df):
    # Create empty df
    _df = pd.DataFrame()
    for nacionality in Ex1_Nacionalidad:
        # Get df filter for given nationality
        filteredDf = df[df[Ex1_Headers.NACIONALIDAD.value] == nacionality.value]
        # Compute sum across columns
        laplaceDf = pd.DataFrame(data=filteredDf.sum(axis = 0, numeric_only = True)).transpose()
        # Apply Laplace to the frequencies given the current class
        laplaceDf = (laplaceDf + 1) / (filteredDf.shape[0] + len(Ex1_Nacionalidad))
        # Append the nationality
        laplaceDf[Ex1_Headers.NACIONALIDAD.value] = nacionality.value
        # Concat the new class probabilities
        _df = pd.concat([_df, laplaceDf], ignore_index = True)
    return _df

def compute_class_probability(df):
    # Create empty df
    _df = pd.DataFrame()
    for nacionality in Ex1_Nacionalidad:
        # Get df filter for given nationality
        filteredDf = df[df[Ex1_Headers.NACIONALIDAD.value] == nacionality.value]
        # Compute sum across columns
        nationalityProbability = pd.DataFrame(data=[[filteredDf.shape[0]/df.shape[0]]])
        # Append the nationality
        nationalityProbability[Ex1_Headers.NACIONALIDAD.value] = nacionality.value
        # Concat the new class probabilities
        _df = pd.concat([_df, nationalityProbability], ignore_index = True)
    return _df

def compute_hmap_for_class(sample, frequencies, class_probability, class_to_calculate):
    # Get the probabilities for just this class
    current_class_frequencies, current_class_probability = get_df_for_class(frequencies, class_to_calculate), get_df_for_class(class_probability, class_to_calculate)
    # Start with the probability of the class itself
    probability = current_class_probability.iloc[0][0]
    for header in Ex1_Headers:
        # Ignore nationality header
        if header != Ex1_Headers.NACIONALIDAD:
            if (sample[header.value][0] == 1):
                probability *= current_class_frequencies.iloc[0][header.value]
            else:
                probability *= (1 - current_class_frequencies.iloc[0][header.value])
    return probability

def apply_bayes(example, df):
    results, total, max, max_nationality = {}, 0, 0, None
    frequencies, class_probability = compute_laplace_frequencies(df), compute_class_probability(df)
    # Calculate the hmap without denominator
    for nationality in Ex1_Nacionalidad:
        results[nationality] = compute_hmap_for_class(example, frequencies, class_probability, nationality.value)
        total += results[nationality]
    # Iterate again to properly compute the probability
    print("Current example to classify...")
    print(example)
    print('')
    for nationality in Ex1_Nacionalidad:
        # Divide by the total
        results[nationality] = results[nationality] / total
        print("Probability of the example to be", nationality.value, "is", results[nationality])
        # Check which one is the maximum one
        if results[nationality] > max:
            max = results[nationality]
            max_nationality = nationality
    print("It's most probable to be", max_nationality.value)

def run_exercise_1(file):
    # Get the data
    df = read_data(file)
    # Build samples to test
    sample_1, sample_2 = build_examples(df.columns)
    # Apply Bayes to both
    print("------------------------")
    apply_bayes(sample_1, df)
    print("------------------------")
    apply_bayes(sample_2, df)