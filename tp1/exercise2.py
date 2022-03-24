import numpy as np
import pandas as pd
from bayes import apply_bayes
from fileHandling import read_data
from newsProcessing import get_key_words, compute_laplace_frequencies, compute_class_probability, build_binary_survey
from constants import Ex2_Mode, Ex2_Headers, Ex2_Categoria
from configurations import Configuration

def get_data_from_matrix(confusion):
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return {'TP':TP, 'TN':TN, 'FP':FP, 'FN':FN}

def get_total_accuracy(confusion):
    TP = 0.0
    events = 0.0
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            events += confusion[i,j]
            if i==j:
                TP += confusion[i,j]
    return TP / events

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

def run_exercise_2(file, mode):
    print('Importing news data...')
    df = read_data(file)

    allowed_categories = [Ex2_Categoria.DEPORTES, Ex2_Categoria.SALUD, Ex2_Categoria.ENTRETENIMIENTO, Ex2_Categoria.ECONOMIA]
    allowed_categories = [e.value for e in allowed_categories]
    df = df[df[Ex2_Headers.CATEGORIA.value].isin(allowed_categories)]
    msk = np.random.rand(len(df)) < 0.9
    train = df[msk]
    test = df[~msk]

    print('Processing training set...')
    key_words = get_key_words(train, allowed_categories, 50, is_analysis=mode == Ex2_Mode.ANALYZE.value)
    if mode == Ex2_Mode.SOLVE.value:
        frequencies = compute_laplace_frequencies(train, key_words, allowed_categories)
        class_probability = compute_class_probability(train, allowed_categories)

        print('Processing testing set...')
        test_df = build_binary_survey(test, key_words)
        total_elements, current_step = test_df.shape[0], 0
        confusion = np.zeros((len(allowed_categories), len(allowed_categories)))

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
            predicted, actual = apply_bayes(test_df.iloc[[index]].reset_index(drop=True), frequencies, class_probability, key_words, Ex2_Headers.CATEGORIA.value, allowed_categories, print_example=Configuration.isVeryVerbose())
            confusion[allowed_categories.index(actual), allowed_categories.index(predicted)] += 1
        print(confusion)
        print("Accuracy: ", get_accuracy(confusion))
        print("Precision: ", get_precision(confusion))
        print("Recall: ", get_recall(confusion))
        print("F1: ", get_F1_score(confusion))
        print("TP Rate: ", get_TP_rate(confusion))
        print("FP Rate: ", get_FP_rate(confusion))