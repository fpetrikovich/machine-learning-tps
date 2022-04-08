from fileHandling import read_csv
from constants import Ex2_Headers, EX2_DIVISION, Ex2_Title_Sentiment, Ex2_Modes
import numpy as np
from functools import reduce
from confusion import get_accuracy, get_precision
from configurations import Configuration
from plotting import plot_confusion_matrix


def show_analysis(df):
    # Get a filter for 1 star reviews
    filter = df[Ex2_Headers.STAR_RATING.value] == 1
    # Filter DF to get 1 star reviews
    one_start_reviews = df[filter]
    print('---------------------------')
    print('Analysis for 1 star reviews')
    # Map text to word count, make that a list, convert to numpy array and get the average
    # One liner to do that for both types
    print('Average word count per review title --> ', np.average(np.array(list(
        map(lambda x: len(x.split(' ')), one_start_reviews[Ex2_Headers.TITLE.value])))))
    print('Average word count per review text --> ', np.average(np.array(list(
        map(lambda x: len(x.split(' ')), one_start_reviews[Ex2_Headers.TEXT.value])))))
    print('Average word count using wordcount --> ',
          np.average(np.array(one_start_reviews[Ex2_Headers.WORDCOUNT.value])))
    print('---------------------------')

# Preprocesses the dataset and makes all required changes


def preprocess_dataset(df):
    df.loc[df[Ex2_Headers.TITLE_SENTIMENT.value] ==
           Ex2_Title_Sentiment.POSITIVE.value, Ex2_Headers.TITLE_SENTIMENT.value] = 1
    df.loc[df[Ex2_Headers.TITLE_SENTIMENT.value] ==
           Ex2_Title_Sentiment.NEGATIVE.value, Ex2_Headers.TITLE_SENTIMENT.value] = 0
    return df


# Accumulates taking into account the neighbor weight
def neighbor_accum_fn(accum, curr):
    if not curr[0] in accum:
        accum[curr[0]] = 0
    accum[curr[0]] += (1 * curr[1])
    return accum


def get_neighbors(neighbors, k, n):
    is_tied, _k = True, k
    while _k < n and is_tied:
        # Gettings the distribution
        distribution = reduce(neighbor_accum_fn, neighbors[:_k], {})
        # Sorting to get the values in order
        distribution = dict(sorted(distribution.items(),
                            key=lambda item: item[1], reverse=True))
        # Get the distribution items
        distribution = list(distribution.items())
        # If there's more than 1 type of class, we can get ties
        if len(distribution) > 1:
            # Means that there is a tie
            if distribution[0][1] == distribution[1][1]:
                is_tied = True
            else:
                # Otherwise just return the first one which should be larger
                return distribution[0][0]
        else:
            # If there was only 1 class, no need to do more analysis
            # Just return the class of the first item
            return distribution[0][0]
        _k += 1


def perform_classification(train, test, k_neighbors, mode):
    total_elements = test.shape[0]
    confusion = np.zeros((5, 5))
    # Split labels from data
    train_data, train_label, test_data, test_label = train[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.SENTIMENT_VALUE.value]].reset_index(drop=True), train[[Ex2_Headers.STAR_RATING.value]].reset_index(
        drop=True), test[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.SENTIMENT_VALUE.value]].reset_index(drop=True), test[[Ex2_Headers.STAR_RATING.value]].reset_index(drop=True)
    for i in range(total_elements):
        distances_with_index = []
        # Get the current example
        example = test_data.iloc[i]
        # Process all differences to get the distance
        example_diff = train_data - example
        # Process the square, then sum it and apply sqrt
        example_diff = ((example_diff**2).sum(axis=1))**.5
        # Add distances & index of example to array
        for j in range(total_elements):
            distances_with_index.append((example_diff.iloc[j], j))
        # Sort all items of the array (it will sort by default by ascending distance)
        sorted_distances_with_index = sorted(distances_with_index)
        # Map to classes
        # It maps to tuples like (class, weight), where weight is 1 or 1/distance**2 depending on the mode
        neighbors = list(map(
            lambda x: (train_label[Ex2_Headers.STAR_RATING.value].iloc[x[1]], 1 if mode == Ex2_Modes.SIMPLE else 1/(x[0]**2)), sorted_distances_with_index))
        # Keep just the classes of those neighbors
        classification = get_neighbors(neighbors, k_neighbors, train.shape[0])
        # Build confusion matrix
        confusion[test_label[Ex2_Headers.STAR_RATING.value].iloc[i] -
                  1, classification - 1] += 1
    # Getting some metrics
    precision = get_precision(confusion)
    accuracy = get_accuracy(confusion)
    print('---------------------------')
    print('Accuracy --> ', accuracy)
    print('Precision --> ', precision)
    if Configuration.isVerbose:
        plot_confusion_matrix(confusion, [str(x + 1) + ' Stars' for x in range(5)])
    return precision, accuracy


def run_exercise_2(filepath, mode, k_neighbors=5, cross_validation_k=None):
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
