from fileHandling import read_data
from constants import Ex1_Headers, Ex1_Nacionalidad
import pandas as pd

# Create examples as DataFrame to have access to columns
def build_examples(columns):
    sample_1 = pd.DataFrame(data=[[1,0,1,1,0,'?']], columns=columns)
    sample_2 = pd.DataFrame(data=[[0,1,1,0,1,'?']], columns=columns)
    return sample_1, sample_2

# Get row for specific nationality class
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
        # Set data as the realtive frequency of each class
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
            # Determine if we need the probability of it being a "positive" or a "negative" example
            # If it's a positive example, just multiply, otherwise use 1 - probability
            if (sample[header.value][0] == 1):
                probability *= current_class_frequencies.iloc[0][header.value]
            else:
                probability *= (1 - current_class_frequencies.iloc[0][header.value])
    return probability

def apply_bayes(example, frequencies, class_probability):
    results, total, max, max_nationality = {}, 0, 0, None
    # Calculate the hmap without denominator
    for nationality in Ex1_Nacionalidad:
        # Compute the result for each nationality
        results[nationality] = compute_hmap_for_class(example, frequencies, class_probability, nationality.value)
        # Add it towards the total so that we get the correct probability
        total += results[nationality]
    # Pretty prints
    print("Current example to classify...")
    print(example)
    print('')
    # Iterate again to properly compute the probability
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
    # Compute probabilities
    frequencies, class_probability = compute_laplace_frequencies(df), compute_class_probability(df)
    # Apply Bayes to both
    print("------------------------")
    apply_bayes(sample_1, frequencies, class_probability)
    print("------------------------")
    apply_bayes(sample_2, frequencies, class_probability)