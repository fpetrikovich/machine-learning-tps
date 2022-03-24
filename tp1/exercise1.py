from fileHandling import read_data
from bayes import apply_bayes
from constants import Ex1_Headers, Ex1_Nacionalidad
import pandas as pd

# Create examples as DataFrame to have access to columns
def build_examples(columns):
    sample_1 = pd.DataFrame(data=[[1,0,1,1,0,'?']], columns=columns)
    sample_2 = pd.DataFrame(data=[[0,1,1,0,1,'?']], columns=columns)
    return sample_1, sample_2

def compute_laplace_frequencies(df):
    # Create empty df
    _df = pd.DataFrame()
    for nationality in Ex1_Nacionalidad:
        # Get df filter for given nationality
        filteredDf = df[df[Ex1_Headers.NACIONALIDAD.value] == nationality.value]
        # Compute sum across columns
        laplaceDf = pd.DataFrame(data=filteredDf.sum(axis = 0, numeric_only = True)).transpose()
        # Apply Laplace to the frequencies given the current class
        laplaceDf = (laplaceDf + 1) / (filteredDf.shape[0] + len(Ex1_Nacionalidad))
        # Append the nationality
        laplaceDf[Ex1_Headers.NACIONALIDAD.value] = nationality.value
        # Concat the new class probabilities
        _df = pd.concat([_df, laplaceDf], ignore_index = True)
    return _df

def compute_class_probability(df):
    # Create empty df
    _df = pd.DataFrame()
    for nationality in Ex1_Nacionalidad:
        # Get df filter for given nationality
        filteredDf = df[df[Ex1_Headers.NACIONALIDAD.value] == nationality.value]
        # Set data as the realtive frequency of each class
        nationalityProbability = pd.DataFrame(data=[[filteredDf.shape[0]/df.shape[0]]])
        # Append the nationality
        nationalityProbability[Ex1_Headers.NACIONALIDAD.value] = nationality.value
        # Concat the new class probabilities
        _df = pd.concat([_df, nationalityProbability], ignore_index = True)
    return _df

def run_exercise_1(file):
    # Get the data
    df = read_data(file)
    # Compute probabilities
    frequencies, class_probability = compute_laplace_frequencies(df), compute_class_probability(df)
    # Build samples to test
    sample_1, sample_2 = build_examples(df.columns)
    # Apply Bayes to both
    print("------------------------")
    apply_bayes(sample_1, frequencies, class_probability, [e.value for e in Ex1_Headers], Ex1_Headers.NACIONALIDAD.value, [e.value for e in Ex1_Nacionalidad])
    print("------------------------")
    apply_bayes(sample_2, frequencies, class_probability, [e.value for e in Ex1_Headers], Ex1_Headers.NACIONALIDAD.value, [e.value for e in Ex1_Nacionalidad])
