import numpy as np
import pandas as pd
from bayes import apply_bayes, prefilter_dfs
from fileHandling import read_data
from newsProcessing import get_key_words, compute_laplace_frequencies, compute_class_probability, build_binary_survey
from constants import Ex2_Mode, Ex2_Headers, Ex2_Categoria
from configurations import Configuration
from plotting import plot_confusion_matrix
import multiprocessing
import matplotlib.pyplot as plt

def get_data_from_matrix(confusion):
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}

def get_error(confusion):
    mistakes = 0
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            if i!=j:
                mistakes += confusion[i,j]
    return mistakes

def get_accuracy(confusion):
    data = get_data_from_matrix(confusion)
    TP = data['TP']
    FP = data['FP']
    TN = data['TN']
    FN = data['FN']
    return (TP+TN)/(TP+TN+FP+FN)

def get_precision(confusion):
    data = get_data_from_matrix(confusion)
    TP = data['TP']
    FP = data['FP']
    return TP / (TP+FP)

def get_recall(confusion):
    data = get_data_from_matrix(confusion)
    TP = data['TP']
    FN = data['FN']
    return TP / (TP+FN)

def get_F1_score(confusion):
    precision = get_precision(confusion)
    recall = get_recall(confusion)
    return 2*precision*recall/(precision+recall)

def get_TP_rate(confusion):
    data = get_data_from_matrix(confusion)
    TP = data['TP']
    FN = data['FN']
    return TP/(TP+FN)

def get_FP_rate(confusion):
    data = get_data_from_matrix(confusion)
    FP = data['FP']
    TN = data['TN']
    return FP/(FP+TN)

def perform_analysis(train, test, mode, word_count, allowed_categories, plot = True, roc = False):
    key_words = get_key_words(train, allowed_categories, word_count, is_analysis=mode == Ex2_Mode.ANALYZE.value)
    if mode == Ex2_Mode.SOLVE.value:
        frequencies = compute_laplace_frequencies(train, key_words, allowed_categories)
        class_probability = compute_class_probability(train, allowed_categories)
        roc_data = []

        print('Processing testing set...')
        test_df = build_binary_survey(test, key_words)
        total_elements, current_step = test_df.shape[0], 0
        confusion = np.zeros((len(allowed_categories), len(allowed_categories)))
        error = 0

        # Get a clean version of the headers, without the class one
        clean_key_words = [x for x in key_words if x if not x == Ex2_Headers.CATEGORIA.value]

        # Properly store memory to avoid extra computation
        memory = prefilter_dfs(frequencies, class_probability, Ex2_Headers.CATEGORIA.value, allowed_categories, clean_key_words)

        # Apply Bayes to every article in testing set
        for index in range(total_elements):
            if index / 25 > current_step:
                current_step += 1
                print('Processing', current_step * 25, 'out of', total_elements)
            # Print the headline for a better reference
            if Configuration.isVerbose():
                print("\n------------------------\n")
                print(test.iloc[index][Ex2_Headers.TITULAR.value], '-->', test.iloc[index][Ex2_Headers.CATEGORIA.value])
                print('')
            # Use as a sample the indexed location
            predicted, actual, results = apply_bayes(test_df.iloc[[index]].reset_index(drop=True), memory, clean_key_words, Ex2_Headers.CATEGORIA.value, allowed_categories, print_example=Configuration.isVeryVerbose())
            confusion[allowed_categories.index(actual), allowed_categories.index(predicted)] += 1
            roc_data.append({'probabilities': results, 'actual_classification': actual})
            error += (1 if predicted != actual else 0)
        if plot:
            plot_confusion_matrix(confusion, allowed_categories)
        accuracy = get_accuracy(confusion)
        print("\n         \t", allowed_categories)
        print("Accuracy: \t", accuracy)
        print("Precision: \t", get_precision(confusion))
        print("Recall: \t", get_recall(confusion))
        print("F1: \t", get_F1_score(confusion))
        print("TP Rate: \t", get_TP_rate(confusion))
        print("FP Rate: \t", get_FP_rate(confusion), "\n")
        if roc:
            draw_roc_curve(roc_data)
        return error, accuracy

def run_cross_validation_iteration(i, elements_per_bin, df, mode, word_count, allowed_categories, results):
    print('Running cross validation with bin number', i)
    test = df.iloc[i*elements_per_bin:(i+1)*elements_per_bin]
    train = df[~df.index.isin(list(test.index.values))]
    error, precision = perform_analysis(train, test, mode, word_count, allowed_categories, plot = False, roc = False)
    results[i] = [error, precision]

def run_cross_validation(df, cross_k, mode, word_count, allowed_categories):
    # Calculate number of elements per bin
    elements_per_bin = int(len(df)/cross_k)
    print("Running cross validation...")
    # Iterate and run method
    manager = multiprocessing.Manager()
    # Need this dictionary due to non-shared memory issues
    return_dict = manager.dict()
    jobs = [0] * cross_k
    # Create multiple jobs
    for i in range(cross_k):
        jobs[i] = multiprocessing.Process(target=run_cross_validation_iteration, args=(i, elements_per_bin, df, mode, word_count, allowed_categories, return_dict))
        jobs[i].start()
    # Join the jobs for the results
    for i in range(len(jobs)):
        jobs[i].join()
    # Calculate some metrics
    values = return_dict.values()
    errors = np.array([x[0] for x in values])
    accuracies = np.array([x[1] for x in values])
    print('Error average -->', np.average(errors, axis=0), 'std -->', np.std(errors, axis=0))
    print('Accuracy average -->', np.average(accuracies, axis=0), 'std -->', np.std(accuracies, axis=0))

def draw_roc_curve(roc_data):
    class_names = list(roc_data[0]['probabilities'].keys())
    roc_curve = {
        category: {"fp_rate": [],"tp_rate": [], "u":[]} for category in class_names
    }
    for u in np.arange(0, 1.1, 0.1):
        for category in class_names:
            FP,FN,TP,TN = 0,0,0,0
            for article_results in roc_data:
                if article_results['probabilities'][category] > u:
                    # Answers "article is of this category"
                    if category == article_results["actual_classification"]:
                        TP += 1 # Correctly predicted positive result
                    else:
                        FP += 1 # Missed category by answering 'positive'
                else:
                    if category == article_results["actual_classification"]:
                        FN += 1 # Missed category by answering 'negative'
                    else:
                        TN += 1 # Correctly predicted negative result
            fp_rate = FP/float(FP+TN)
            tp_rate = TP/float(TP+FN)
            roc_curve[category]["fp_rate"].append(fp_rate)
            roc_curve[category]["tp_rate"].append(tp_rate)
            roc_curve[category]["u"].append(u)
    for category in class_names:
        plt.plot(roc_curve[category]["fp_rate"], roc_curve[category]["tp_rate"], "-o", label=category)
        if Configuration.isVerbose():
            for i in range(len(roc_curve[category]["fp_rate"])):
                u = int(roc_curve[category]["u"][i]*10)/10.0
                if u*10 % 5 == 0:
                    plt.text(roc_curve[category]["fp_rate"][i]+0.005, roc_curve[category]["tp_rate"][i]-0.005, u)
    line = [x for x in np.arange(0, 1.1, 0.1)]
    plt.plot(line, line, "--", color="grey")
    ax = plt.gca()
    ax.set_xlim([-0.05,1.05])
    ax.set_ylim([-0.05,1.05])
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.legend()
    plt.show()

def run_exercise_2(file, mode, word_count, cross_k = None, roc = False):
    print('Importing news data...')
    df = read_data(file)

    allowed_categories = [  Ex2_Categoria.DEPORTES,
                            Ex2_Categoria.SALUD,
                            Ex2_Categoria.ENTRETENIMIENTO,
                            Ex2_Categoria.ECONOMIA,
                            Ex2_Categoria.CIENCIA_TECNOLOGIA,
                            Ex2_Categoria.NACIONAL,
                            Ex2_Categoria.INTERNACIONAL,
                            ]
    allowed_categories = [e.value for e in allowed_categories]
    df = df[df[Ex2_Headers.CATEGORIA.value].isin(allowed_categories)]
    # Shuffle the DF
    df = df.sample(frac=1)

    print('Processing training set...')
    if mode == Ex2_Mode.SOLVE.value:
        # No cross validation, just use 90% as train and 10% as test
        if cross_k == None:
            train_number = int(len(df)/10) * 9
            train = df.iloc[0:train_number]
            test = df.iloc[train_number+1:len(df)]
            perform_analysis(train, test, mode, word_count, allowed_categories, roc=roc)
        else:
            run_cross_validation(df, cross_k, mode, word_count, allowed_categories)
    else:
        print(df.iloc[10:20])
        # Use the 90% as train
        train_number = int(len(df)/10) * 9
        train = df.iloc[0:train_number]
        test = df.iloc[train_number+1:len(df)]
        get_key_words(train, allowed_categories, word_count, is_analysis=mode == Ex2_Mode.ANALYZE.value)
