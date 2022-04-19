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
from tree import export_trees
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

def build_random_forest_trees(df, train_samples, should_count = False, new_height = None):
    example_umbral = Parameters.get_example_umbral()
    gain_umbral = Parameters.get_gain_umbral()
    height_limit = Parameters.get_height_limit() if new_height is None else new_height
    split_attr_limit = Parameters.get_split_attr_limit()
    nodes_count = 0

    trees = []
    for train_sample in train_samples:
        tree = make_tree(df, train_sample, Ex1_Headers.CREDITABILITY.value, example_umbral, gain_umbral, height_limit, split_attr_limit)
        trees.append(tree)
        if should_count:
            nodes_count += count_tree_nodes(tree)
    return trees, nodes_count

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

    m = stats.mode(np.asarray(tree_classifications, dtype=object))
    # print(m)
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
    # print('---------------------------')
    # print('Error --> ', error, '\nAccuracy --> ',
    #       accuracy, '\nPrecision --> ', precision)
    # print('---------------------------')
    if Configuration.isVerbose() or show_matrix:
        plot_confusion_matrix(confusion, ["REJECTED", "APPROVED"])
    return error, accuracy, precision

def run_rf_cross_validation_iteration(i, elements_per_bin, df, goal_attribute, sample_size, number_sample, results, is_analysis = False, analytical_height = None):
    test = df.iloc[i*elements_per_bin:(i+1)*elements_per_bin]
    train = df[~df.index.isin(list(test.index.values))]
    bootstrapped_datasets = bootstrap_dataset(train, sample_size, number_sample)
    trees, node_count = build_random_forest_trees(df, bootstrapped_datasets, is_analysis, analytical_height)
    # if i == 11: export_trees(trees)
    final_classifications = perform_rf_classification(trees, test)
    error, accuracy, precision = analyze_results(train, test, final_classifications, goal_attribute)
    # Also analyse the train set results if is_analysis
    if is_analysis:
        final_classifications_train = perform_rf_classification(trees, train)
        error_train, accuracy_train, precision_train = analyze_results(train, train, final_classifications_train, goal_attribute)
        results[i] = [error, accuracy, precision, error_train, accuracy_train, precision_train, node_count]
    else:
        results[i] = [error, accuracy, precision]

def run_rf_cross_validation(df, cross_k, sample_size, number_sample, is_analysis = False, analytical_height = None):
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
        jobs[i] = multiprocessing.Process(target=run_rf_cross_validation_iteration, args=(i, elements_per_bin, df, Ex1_Headers.CREDITABILITY.value, sample_size, number_sample, return_dict, is_analysis, analytical_height))
        jobs[i].start()
    # Join the jobs for the results
    for i in range(len(jobs)):
        jobs[i].join()
    # Calculate some metrics
    values = return_dict.values()
    errors_test = np.array([x[0] for x in values])
    accuracies_test = np.array([x[1] for x in values])
    avg_error_test = np.average(errors_test, axis=0)/elements_per_bin 
    # print('---------------------------')
    # print('---------------------------')
    # print('---------------------------')
    # print('Error average -->', avg_error_test, '\nstd -->', np.std(errors_test, axis=0))
    # print('Accuracy average -->', np.average(accuracies_test, axis=0), '\nstd -->', np.std(accuracies_test, axis=0))
    
    # Also show the results for train precision if in analysis
    if is_analysis:
        errors_train = np.array([x[3] for x in values])
        nodes = np.array([x[6] for x in values])
        avg_error_train = np.average(errors_train, axis=0)/(elements_per_bin * (cross_k - 1))
        avg_node_count = np.average(nodes)
        return 1 - avg_error_train, 1 - avg_error_test, int(avg_node_count)
    
    # Return the percentage of correct classifications
    return 1 - avg_error_test

def run_exercise_1_rf(filepath, cross_validation_k=None, mode=Ex2_Run.SOLVE):
    df = read_csv(filepath, ',')
    df = dicretize_data(df)
    # print_entire_df(df)
    sample_size = Parameters.get_sample_size()
    number_sample = Parameters.get_number_samples()

    # Shuffle df
    df = df.sample(frac=1)

    if mode == Ex2_Run.ANALYZE:
        if cross_validation_k == None: return
        show_rf_analysis(df, cross_validation_k, sample_size, number_sample)
        return

    if cross_validation_k == None:
        # Divide dataset
        train_number = int(len(df)/EX1_DIVISION) * (EX1_DIVISION - 1)
        train = df.iloc[0:train_number]
        test = df.iloc[train_number+1:len(df)]
        bootstrapped_datasets = bootstrap_dataset(train, sample_size, number_sample)

        print("Started building trees at", datetime.datetime.now())
        trees, _ = build_random_forest_trees(df, bootstrapped_datasets)
        print("Finished building trees at", datetime.datetime.now())
        final_classifications = perform_rf_classification(trees, test)
        export_trees(trees)
        analyze_results(train, test, final_classifications, Ex1_Headers.CREDITABILITY.value, True)
    else:
        run_rf_cross_validation(df, cross_validation_k, sample_size, number_sample)

def show_rf_analysis(df, cross_k, sample_size, number_sample):
    df = df.sample(frac=1)
    nodes = []
    train_precisions = []
    test_precisions = []
    for max_height in range(8):
        print("h=", max_height)
        train_precision, test_precision, node_count = run_rf_cross_validation(df, cross_k, sample_size, number_sample, True, max_height)
        print(train_precision, test_precision, node_count)
        nodes.append(node_count)
        train_precisions.append(train_precision)
        test_precisions.append(test_precision)
    plt.plot(nodes, train_precisions, label = "Training Set Precision", marker='o')
    plt.plot(nodes, test_precisions, label = "Testing Set Precision", marker='o')
    for i in range(len(test_precisions)):
        x = nodes[i]
        y = test_precisions[i]
        if i<2 or (i>1 and x != nodes[i-1]):
            plt.text(x * (1 + 0.01), y * (1 + 0.01), "h="+str(i))
    plt.xlabel('Cantidad de Nodos')
    plt.ylabel('Presición')
    plt.ylim(0.5,1.05)
    plt.legend()
    plt.show()
    # Redo for best one, save the matrix this creates
    best_height = test_precisions.index(max(test_precisions))
    print("Best height was", best_height)
