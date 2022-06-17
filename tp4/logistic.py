# External
from functools import reduce
import statsmodels.formula.api as smf
from datetime import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Local
from config.constants import Headers
from fileHandling import split_dataset, split_dataset_data_labels, shuffle_dataset, print_entire_df
from utils.confusion import build_and_save_confusion_matrix

TRAIN_TEST = 0.75
RESULTS_DIR = './results'

####################################################################################
##################################### RESULTS ######################################
####################################################################################

def create_results_if_not_exist():
    # https://appdividend.com/2021/07/03/how-to-create-directory-if-not-exist-in-python/#:~:text=To%20create%20a%20directory%20if,makedirs()%20method.
    path = RESULTS_DIR
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist 
        os.makedirs(path)
        print("[INFO] Results directory was created!")

####################################################################################
##################################### EXAMPLE ######################################
####################################################################################

def get_sample_test():
    d = {Headers.AGE.value: [60], Headers.CAD_DUR_GOOD.value: [2], Headers.CHOLESTEROL.value: [199]}
    df = pd.DataFrame(data=d)
    return df

####################################################################################
###################################### STATS #######################################
####################################################################################

def model_stats(clf):
    print("\n\n")
    print(clf.summary())
    print("\n\n")

def manually_calculate_model(coefficients, values):
    _exp = np.exp(np.sum(coefficients * values))
    return _exp / (1 + _exp)

####################################################################################
##################################### HEADERS ######################################
####################################################################################

def get_headers(account_male_female = False):
    predictor_headers = [Headers.AGE.value, Headers.CAD_DUR_GOOD.value, Headers.CHOLESTEROL.value]
    result_headers = [Headers.SIGDZ.value]
    if account_male_female:
        predictor_headers.append(Headers.SEX.value)
    return predictor_headers, result_headers

def modify_dataset(df):
    df[Headers.CAD_DUR_GOOD.value] = df[Headers.CAD_DUR.value]
    return df

####################################################################################
###################################### MODEL #######################################
####################################################################################

def get_model(data, predictor_headers, outcome_headers):
    left_side = reduce(lambda a,b: a + ' + ' + b, outcome_headers)
    right_side = reduce(lambda a,b: a + ' + ' + b, predictor_headers)
    formula = left_side + " ~ " + right_side
    clf = smf.logit(formula=formula, data = data).fit()
    return clf

def use_model(clf, data):
    tmp_predictions = clf.predict(data)
    predictions = np.array(list(map(round, tmp_predictions)))
    return predictions

def run_logistic(df, cross_k = None, account_male_female = False):
    # Create result directory if not exists
    create_results_if_not_exist()

    # Shuffle dataset
    df = shuffle_dataset(df)
    # Modify the headings so that they are properly named
    df = modify_dataset(df)
    # Get headers to use later
    predictor_headers, result_headers = get_headers(account_male_female)

    # Split dataset
    train, test = split_dataset(df, TRAIN_TEST)

    # Get data and labels division
    train_data, train_labels = split_dataset_data_labels(train, predictor_headers, result_headers)
    train_labels = train_labels.values.ravel()
    test_data, test_labels = split_dataset_data_labels(test, predictor_headers, result_headers)
    test_labels = test_labels.values.ravel()
    # Get the logistics model
    # We have to ravel in order to get a proper array if labels
    clf = get_model(train, predictor_headers, result_headers)
    # Making predictions
    predictions = use_model(clf, test_data)

    # Confusion
    result_filepath = f'{RESULTS_DIR}/confusion_{int(datetime.timestamp(datetime.now()))}.png'
    print(f'[INFO] Building and saving confusion matrix ({result_filepath})...')
    confusion = build_and_save_confusion_matrix(predictions, test_labels, (2,2), ['0', '1'], result_filepath)
    
    # PValue
    model_stats(clf)

    # Accuracy
    precision = confusion[1,1] / (confusion[1,1] + confusion[0,1])
    accuracy = (confusion[1,1] + confusion[0,0]) / (np.sum(confusion))
    print(f'[RESULT] Model accuracy is {accuracy}')
    print(f'[RESULT] Model precision is {precision}')
    print("\n\n")

    # Testing the given sample
    sample_df = get_sample_test()
    print(f'[INFO] Testing sample...')
    print(sample_df)
    basic_values = np.array([1])
    if account_male_female:
        # Keep the basic value, which is just the intercept coefficient
        male_values, female_values = np.copy(basic_values), np.copy(basic_values)
        # Add the values
        male_values = np.append(male_values, sample_df.values[0])
        female_values = np.append(female_values, sample_df.values[0])
        # Add male/female tag
        male_values = np.append(male_values, 0)
        female_values = np.append(female_values, 1)
        # Calculate probability
        male_prob = manually_calculate_model(clf.params.values, male_values)
        female_prob = manually_calculate_model(clf.params.values, female_values)
        print(f'[RESULT] Probability if person is male is {male_prob} ==> p >= 0.5 ==> {np.round(male_prob)}')
        print(f'[RESULT] Probability if person is female is {female_prob} ==> p >= 0.5 ==> {np.round(female_prob)}')
    else:
        basic_values = np.append(basic_values, sample_df.values[0])
        prob = manually_calculate_model(clf.params.values, basic_values)
        print(f'[RESULT] Probability is {prob} ==> p >= 0.5 ==> {np.round(prob)}')