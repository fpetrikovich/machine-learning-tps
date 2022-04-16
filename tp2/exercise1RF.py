from fileHandling import read_csv, print_entire_df
from constants import Ex1_Headers, EX1_DIVISION, Ex2_Modes, Ex2_Run
from confusion import get_accuracy, get_precision
from parameters import Parameters
from plotting import plot_confusion_matrix
from configurations import Configuration
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import datetime
from scipy import stats
from tree import export_tree
from exercise1 import *

def bootstrap_dataset(df, sample_size, number_sample):
    """ Example of bootstrapping a dataset.
        Bootstrapping takes N samples of a given size, with replacement, from a dataset.
        Returns an array with all bootstrapped samples.
        :param dataset: dataset to bootstrap
        :param sample_size: percentage of the dataset we want for each sample
        :param number_samples: total number of samples to bootstrap
    """
    bootstrap_datasets = []

    for _sample in range(0, number_sample):
        # selecting with replacement
        sampled_df = df.sample(frac=sample_size, replace=True)
        bootstrap_datasets.append(sampled_df)

    return bootstrap_datasets

def build_random_forest_trees(df, train_samples):
    example_umbral = Parameters.get_example_umbral()
    gain_umbral = Parameters.get_gain_umbral()
    height_limit = Parameters.get_height_limit()
    split_attr_limit = Parameters.get_split_attr_limit()
    
    trees = []
    for train_sample in train_samples:
        tree = make_tree(df, train_sample, Ex1_Headers.CREDITABILITY.value, example_umbral, gain_umbral, height_limit, split_attr_limit)
        trees.append(tree)
    
    return trees

def perform_single_tree_classification(tree, test_sample):
    """ Recieve the tree and the test sample and returns an array of the predicted
        classifications using said tree.
        :param tree: tree to use for predictions
        :param test_sample: data to analyze and predict
    """
    total_test_elements = test_sample.shape[0]
    classifications = []

    for i in range(total_test_elements):
        entry = test_sample.iloc[i]
        prediction = classify(entry, tree)
        classifications.append(prediction)

    return classifications

def perform_rf_classification(trees, test_sample):
    tree_classifications = []
    for tree in trees:
        classification = perform_single_tree_classification(tree, test_sample)
        tree_classifications.append(classification)
    m = stats.mode(np.array(tree_classifications))
    print(m)
    return m[0][0]

def analyze_results(train_sample, test_sample, predictions, goal_attribute, show_matrix=False):
    total_train_elements, total_test_elements = train_sample.shape[0], test_sample.shape[0]
    possible_answers = train_sample[goal_attribute].unique()

    error = 0
    confusion = np.zeros((len(possible_answers), len(possible_answers)))

    for i in range(total_test_elements):
        entry = test_sample.iloc[i]
        actual = entry[goal_attribute]
        prediction = predictions[i]
        confusion[actual][prediction] += 1
        if actual != prediction:
            error += 1
    precision = get_precision(confusion)
    accuracy = get_accuracy(confusion)
    print('---------------------------')
    print('Error --> ', error, '\nAccuracy --> ',
          accuracy, '\nPrecision --> ', precision)
    print('---------------------------')
    if Configuration.isVerbose() or show_matrix:
        plot_confusion_matrix(confusion, ["REJECTED", "APPROVED"])
    return error, accuracy, precision

def run_exercise_1_ft(filepath, cross_validation_k=None, mode=Ex2_Run.SOLVE):
    df = read_csv(filepath, ',')
    df = dicretize_data(df)
    # print_entire_df(df)
    sample_size = Parameters.get_sample_size()
    number_sample = Parameters.get_number_samples()

    # Shuffle df
    df = df.sample(frac=1)

    if cross_validation_k == None:
        # Divide dataset
        train_number = int(len(df)/EX1_DIVISION) * (EX1_DIVISION - 1)
        train = df.iloc[0:train_number]
        test = df.iloc[train_number+1:len(df)]
        bootstrapped_datasets = bootstrap_dataset(train, sample_size, number_sample)

        print("Started building trees at", datetime.datetime.now())
        trees = build_random_forest_trees(df, bootstrapped_datasets)
        print("Finished building trees at", datetime.datetime.now())
        final_classifications = perform_rf_classification(trees, test)
        export_tree(trees[0])
        analyze_results(train, test, final_classifications, Ex1_Headers.CREDITABILITY.value, True)
    else:
        run_cross_validation(df=df, cross_k=cross_validation_k)
