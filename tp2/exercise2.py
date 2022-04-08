from fileHandling import read_csv, print_entire_df
from constants import Ex2_Headers, EX2_DIVISION, Ex2_Title_Sentiment
import numpy as np

def show_analysis(df):
    # Get a filter for 1 star reviews
    filter = df[Ex2_Headers.STAR_RATING.value] == 1
    # Filter DF to get 1 star reviews
    one_start_reviews = df[filter]
    print('---------------------------')
    print('Analysis for 1 star reviews')
    # Map text to word count, make that a list, convert to numpy array and get the average
    # One liner to do that for both types
    print('Average word count per review title --> ', np.average(np.array(list(map(lambda x: len(x.split(' ')), one_start_reviews[Ex2_Headers.TITLE.value])))))
    print('Average word count per review text --> ', np.average(np.array(list(map(lambda x: len(x.split(' ')), one_start_reviews[Ex2_Headers.TEXT.value])))))
    print('Average word count using wordcount --> ', np.average(np.array(one_start_reviews[Ex2_Headers.WORDCOUNT.value])))
    print('---------------------------')

# Preprocesses the dataset and makes all required changes
def preprocess_dataset(df):
    df.loc[df[Ex2_Headers.TITLE_SENTIMENT.value] == Ex2_Title_Sentiment.POSITIVE.value, Ex2_Headers.TITLE_SENTIMENT.value] = 1
    df.loc[df[Ex2_Headers.TITLE_SENTIMENT.value] == Ex2_Title_Sentiment.NEGATIVE.value, Ex2_Headers.TITLE_SENTIMENT.value] = 0
    return df

def perform_classification(train, test, k_neighbors, mode):
    total_element = test.shape[0]
    # Split labels from data
    train_data, train_label, test_data, test_label = train[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.SENTIMENT_VALUE.value]].reset_index(drop=True), train[[Ex2_Headers.STAR_RATING.value]].reset_index(drop=True), test[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.SENTIMENT_VALUE.value]].reset_index(drop=True), test[[Ex2_Headers.STAR_RATING.value]].reset_index(drop=True)
    for i in range(total_element):
        distances_with_index = []
        # Get the current example
        example = test_data.iloc[i]
        # Process all differences to get the distance
        example_diff = train_data - example
        # Process the square, then sum it and apply sqrt
        example_diff = ((example_diff**2).sum(axis=1))**.5
        # Add distances & index of example to array
        for j in range(total_element):
            distances_with_index.append((example_diff.iloc[j], j))
        # Sort all items of the array (it will sort by default by ascending distance)
        sorted_distances_with_index = sorted(distances_with_index)
        print(sorted_distances_with_index)

def run_exercise_2(filepath, mode, k_neighbors = 5, cross_validation_k = None):
    df = read_csv(filepath, ";")
    # Perform the analysis requested
    show_analysis(df)
    # print_entire_df(df)
    # TODO: clean dataset
    # Shuffle df
    df = df.sample(frac=1)
    df = preprocess_dataset(df)
    # TODO: Perform test & train division
    # No cross validation scenario
    if cross_validation_k == None:
        # Divide dataset
        train_number = int(len(df)/EX2_DIVISION) * (EX2_DIVISION - 1)
        train = df.iloc[0:train_number]
        test = df.iloc[train_number+1:len(df)]
        perform_classification(train, test, k_neighbors=k_neighbors, mode=mode)
    # Apply cross validation with different processes
    else:
        a = 2