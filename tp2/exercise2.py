from math import nan
from fileHandling import read_csv, print_entire_df
from constants import Ex2_Headers, EX2_DIVISION, Ex2_Title_Sentiment, Ex2_Modes, Ex2_Run
import numpy as np
from functools import reduce
from confusion import get_accuracy, get_precision
from configurations import Configuration
from plotting import plot_confusion_matrix
import multiprocessing

# Hide possible 0/0 warnings
np.seterr(invalid='ignore')


def show_analysis(df):
    print('\n###########################')
    print('ANALYSIS - START')
    print('###########################\n')
    print('Analysis for 1 star reviews')
    # Get a filter for 1 star reviews
    filter = df[Ex2_Headers.STAR_RATING.value] == 1
    # Filter DF to get 1 star reviews
    one_start_reviews = df[filter]
    # Map text to word count, make that a list, convert to numpy array and get the average
    # One liner to do that for both types
    print('Average word count per review title --> ', np.average(np.array(list(
        map(lambda x: len(x.split(' ')), one_start_reviews[Ex2_Headers.TITLE.value])))))
    print('Average word count per review text --> ', np.average(np.array(list(
        map(lambda x: len(x.split(' ')), one_start_reviews[Ex2_Headers.TEXT.value])))))
    print('Average word count using wordcount --> ',
          np.average(np.array(one_start_reviews[Ex2_Headers.WORDCOUNT.value])))
    print('\n---------------------------\n')
    print('Word count per star review')
    for i in range(5):
        filter = df[Ex2_Headers.STAR_RATING.value] == (i + 1)
        filtered_df = df[filter]
        filtered_wordcount = np.array(filtered_df[Ex2_Headers.WORDCOUNT.value])
        print(f'{i + 1} Stars word average --> {np.average(filtered_wordcount)}')
        print(f'{i + 1} Stars word std --> {np.std(filtered_wordcount)}')
    print('\n---------------------------\n')
    print('Review frequency')
    frecuencies = [df[df[Ex2_Headers.STAR_RATING.value] == x + 1].shape[0]/df.shape[0] for x in range(5)]
    for i in range(5):
        print(i + 1, 'Stars frecuency -->', frecuencies[i], f'({int(frecuencies[i] * df.shape[0])})')
    print('\n---------------------------\n')
    print('Reviews with non-matching text/title sentiment')
    non_matching_sentiment_df = df[df[Ex2_Headers.TITLE_SENTIMENT.value] != df[Ex2_Headers.TEXT_SENTIMENT.value]]
    print('Number of entries with non-matching sentiment in review -->', non_matching_sentiment_df.shape[0])
    print('Relative frequency -->', non_matching_sentiment_df.shape[0] / df.shape[0])
    frecuencies = [non_matching_sentiment_df[non_matching_sentiment_df[Ex2_Headers.STAR_RATING.value] == x + 1].shape[0] for x in range(5)]
    for i in range(5):
        print(f'{i + 1} Stars frequency --> {frecuencies[i]/non_matching_sentiment_df.shape[0]} ({frecuencies[i]} occurences)')
    print('\n---------------------------\n')
    print('Entries with missing data')
    empty_filter = df[Ex2_Headers.TITLE_SENTIMENT.value].notnull()
    empty_df = df[~empty_filter]
    print(f'Total number of rows that contain missing Title Sentiment --> {empty_df.shape[0]}')
    for i in range(5):
        filter = empty_df[Ex2_Headers.STAR_RATING.value] == (i + 1)
        filtered_df = empty_df[filter]
        print(f'{i + 1} Stars with empty data --> {filtered_df.shape[0]}')
    print('\n###########################')
    print('ANALYSIS - END')
    print('###########################\n')

# Preprocesses the dataset and makes all required changes


def preprocess_dataset(df):
    df.loc[df[Ex2_Headers.TITLE_SENTIMENT.value] ==
           Ex2_Title_Sentiment.POSITIVE.value, Ex2_Headers.TITLE_SENTIMENT.value] = 1
    df.loc[df[Ex2_Headers.TITLE_SENTIMENT.value] ==
           Ex2_Title_Sentiment.NEGATIVE.value, Ex2_Headers.TITLE_SENTIMENT.value] = 0
    df.loc[df[Ex2_Headers.TEXT_SENTIMENT.value] ==
           Ex2_Title_Sentiment.POSITIVE.value, Ex2_Headers.TEXT_SENTIMENT.value] = 1
    df.loc[df[Ex2_Headers.TEXT_SENTIMENT.value] ==
           Ex2_Title_Sentiment.NEGATIVE.value, Ex2_Headers.TEXT_SENTIMENT.value] = 0
    # Normalize
    temp_df = df[Ex2_Headers.WORDCOUNT.value]
    df[Ex2_Headers.WORDCOUNT.value] = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())
    temp_df = df[Ex2_Headers.SENTIMENT_VALUE.value]
    df[Ex2_Headers.SENTIMENT_VALUE.value] = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())
    temp_df = df[Ex2_Headers.STAR_RATING.value]
    df[Ex2_Headers.STAR_RATING_NORM.value] = (temp_df - temp_df.min()) / (temp_df.max() - temp_df.min())
    df[Ex2_Headers.ORIGINAL_ID.value] = np.arange(0, df.shape[0])
    return df

# Replaces the missing column data based on the given replacement information
def perform_replacements(df, replacements):
    for replace in replacements:
        _to, _from = replace[0], replace[1]
        df.loc[_to, Ex2_Headers.TITLE_SENTIMENT.value] = df.iloc[_from][Ex2_Headers.TITLE_SENTIMENT.value]
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
    total_train_elements, total_test_elements = train.shape[0], test.shape[0]
    confusion = np.zeros((5, 5))
    error = 0
    # Split labels from data
    train_data, train_label, test_data, test_label = train[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.SENTIMENT_VALUE.value]].reset_index(drop=True), train[[Ex2_Headers.STAR_RATING.value]].reset_index(
        drop=True), test[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.SENTIMENT_VALUE.value]].reset_index(drop=True), test[[Ex2_Headers.STAR_RATING.value]].reset_index(drop=True)
    for i in range(total_test_elements):
        distances_with_index = []
        # Get the current example
        example = test_data.iloc[i]
        # Process all differences to get the distance
        example_diff = train_data - example
        # Process the square, then sum it and apply sqrt
        example_diff = ((example_diff**2).sum(axis=1))**.5
        # Add distances & index of example to array
        for j in range(total_train_elements):
            distances_with_index.append((example_diff.iloc[j], j))
        # Sort all items of the array (it will sort by default by ascending distance)
        sorted_distances_with_index = sorted(distances_with_index)
        # Map to classes
        # It maps to tuples like (class, weight), where weight is 1 or 1/distance**2 depending on the mode
        neighbors = list(map(
            lambda x: (train_label[Ex2_Headers.STAR_RATING.value].iloc[x[1]], 1 if mode == Ex2_Modes.SIMPLE else 1/((x[0] if x[0] > 0 else 1)**2)), sorted_distances_with_index))
        # Keep just the classes of those neighbors
        classification = get_neighbors(neighbors, k_neighbors, train.shape[0])
        # Build confusion matrix
        confusion[test_label[Ex2_Headers.STAR_RATING.value].iloc[i] -
                  1, classification - 1] += 1
        # Add to error
        error += 1 if test_label[Ex2_Headers.STAR_RATING.value].iloc[i] != classification else 0
    # Getting some metrics
    precision = get_precision(confusion)
    accuracy = get_accuracy(confusion)
    print('---------------------------')
    print('Error --> ', error, '\nAccuracy --> ',
          accuracy, '\nPrecision --> ', precision)
    print('---------------------------')
    if Configuration.isVerbose():
        plot_confusion_matrix(
            confusion, [str(x + 1) + ' Stars' for x in range(5)])
    return error, accuracy, precision


def perform_reduced_classification(train, test, k_neighbors, print_results = False):
    total_train_elements, total_test_elements = train.shape[0], test.shape[0]
    # Split labels from data
    train_data, test_data = train[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.SENTIMENT_VALUE.value, Ex2_Headers.STAR_RATING_NORM.value]].reset_index(
        drop=True), test[[Ex2_Headers.WORDCOUNT.value, Ex2_Headers.SENTIMENT_VALUE.value, Ex2_Headers.STAR_RATING_NORM.value]].reset_index(drop=True)
    full_train_data, full_test_data = train[[Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.WORDCOUNT.value, Ex2_Headers.SENTIMENT_VALUE.value, Ex2_Headers.STAR_RATING.value, Ex2_Headers.ORIGINAL_ID.value]].reset_index(
        drop=True), test[[Ex2_Headers.TITLE_SENTIMENT.value, Ex2_Headers.WORDCOUNT.value, Ex2_Headers.SENTIMENT_VALUE.value, Ex2_Headers.STAR_RATING.value, Ex2_Headers.ORIGINAL_ID.value]].reset_index(drop=True)
    # Result
    result = []
    for i in range(total_test_elements):
        distances_with_index = []
        # Get the current example
        example = test_data.iloc[i]
        # Process all differences to get the distance
        example_diff = train_data - example
        # Process the square, then sum it and apply sqrt
        example_diff = ((example_diff**2).sum(axis=1))**.5
        # Add distances & index of example to array
        for j in range(total_train_elements):
            distances_with_index.append((example_diff.iloc[j], j))
        # Sort all items of the array (it will sort by default by ascending distance)
        # Get the nearest k neighbors requested
        sorted_distances_with_index = sorted(
            distances_with_index)[:k_neighbors]
        # Map to classes
        # It maps to tuples like (class, weight), where weight is 1 or 1/distance**2 depending on the mode
        neighbors = list(
            map(lambda x: (full_train_data.iloc[x[1]], x[0]), sorted_distances_with_index))
        if print_results:
            print('----------')
            print("----Point to test----")
            print(full_test_data.iloc[i])
            print("----Neighbors----")
            for n in neighbors:
                print(n)
            print('----------')
        # Just get the first neighbor so that we can replace
        result.append((full_test_data.iloc[i][Ex2_Headers.ORIGINAL_ID.value], neighbors[0][0][Ex2_Headers.ORIGINAL_ID.value]))
    return result

def analyze_fill_data(df, print_results):
    # Run some preprocessing
    df = preprocess_dataset(df)
    filter = df[Ex2_Headers.TITLE_SENTIMENT.value].notnull()
    # Try to complete with the 1 neighbor
    # No shuffle for this dataset
    train = df[filter]
    test = df[~filter]
    # Run classification to find nearest neighbor for each NaN one
    replacements = perform_reduced_classification(train, test, k_neighbors=1, print_results=print_results)
    df = perform_replacements(df, replacements)
    return df

def run_cross_validation_iteration(i, elements_per_bin, df, k_neighbors, mode, results):
    print('Running cross validation with bin number', i)
    test = df.iloc[i*elements_per_bin:(i+1)*elements_per_bin]
    train = df[~df.index.isin(list(test.index.values))]
    error, accuracy, precision = perform_classification(
        train, test, k_neighbors=k_neighbors, mode=mode)
    results[i] = [error, accuracy, precision]


def run_cross_validation(df, cross_k, k_neighbors, mode):
    # Calculate number of elements per bin
    elements_per_bin = int(len(df)/cross_k)
    print("Running cross validation using", cross_k,
          "bins with", elements_per_bin, "elements per bin")
    # Iterate and run method
    manager = multiprocessing.Manager()
    # Need this dictionary due to non-shared memory issues
    return_dict = manager.dict()
    jobs = [0] * cross_k
    # Create multiple jobs
    for i in range(cross_k):
        jobs[i] = multiprocessing.Process(target=run_cross_validation_iteration, args=(
            i, elements_per_bin, df, k_neighbors, mode, return_dict))
        jobs[i].start()
    # Join the jobs for the results
    for i in range(len(jobs)):
        jobs[i].join()
    # Calculate some metrics
    values = return_dict.values()
    errors = np.array([x[0] for x in values])
    accuracies = np.array([x[1] for x in values])
    print('---------------------------')
    print('---------------------------')
    print('---------------------------')
    print('Error average -->', np.average(errors, axis=0),
          '\nstd -->', np.std(errors, axis=0))
    print('Accuracy average -->', np.average(accuracies, axis=0),
          '\nstd -->', np.std(accuracies, axis=0))


def run_exercise_2(filepath, mode, k_neighbors=5, cross_validation_k=None, solve_mode=Ex2_Run.SOLVE):
    df = read_csv(filepath, ";")
    if solve_mode == Ex2_Run.ANALYZE:
        # Perform the analysis requested
        show_analysis(df)
        # Print DF if this is verbose
        if Configuration.isVerbose():
            print_entire_df(df)
        # Analyze to fill missing data
        df = analyze_fill_data(df, print_results=True)
        print('---------------------------')
        print('---------------------------')
        print('AFTER MAKING REPLACEMENTS')
        print('---------------------------')
        print('---------------------------')
        # Perform the analysis requested
        show_analysis(df)
    else:
        # Analyze to fill missing data
        df = analyze_fill_data(df, print_results=Configuration.isVeryVerbose())
        # Shuffle df
        df = df.sample(frac=1)
        df = preprocess_dataset(df)
        # No cross validation scenario
        if cross_validation_k == None:
            # Divide dataset
            train_number = int(len(df)/EX2_DIVISION) * (EX2_DIVISION - 1)
            train = df.iloc[0:train_number]
            test = df.iloc[train_number+1:len(df)]
            perform_classification(
                train, test, k_neighbors=k_neighbors, mode=mode)
        # Apply cross validation with different processes
        else:
            run_cross_validation(
                df=df, cross_k=cross_validation_k, k_neighbors=k_neighbors, mode=mode)
