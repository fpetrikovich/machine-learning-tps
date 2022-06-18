# External
from functools import reduce
import statsmodels.formula.api as smf
from datetime import datetime
import os
import numpy as np
import pandas as pd
import multiprocessing

# Local
from config.constants import Headers
from fileHandling import split_dataset, split_dataset_data_labels, shuffle_dataset, print_entire_df
from config.best_separation import BestSeparation
from utils.confusion import build_and_save_confusion_matrix, build_confusion_matrix

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
#################################### RUN NORMAL ####################################
####################################################################################

def dataset_train_test(train, test, predictor_headers, result_headers):
    # Get data and labels division
    train_data, train_labels = split_dataset_data_labels(train, predictor_headers, result_headers)
    train_labels = train_labels.values.ravel()
    test_data, test_labels = split_dataset_data_labels(test, predictor_headers, result_headers)
    test_labels = test_labels.values.ravel()
    return train, test, train_data, train_labels, test_data, test_labels

def perform_train_test(train, test_data, test_labels, predictor_headers, result_headers, verbose = True):
    # Get the logistics model
    # We have to ravel in order to get a proper array if labels
    clf = get_model(train, predictor_headers, result_headers, disp = verbose)
    # Making predictions
    predictions = use_model(clf, test_data)

    # Confusion
    result_filepath = f'{RESULTS_DIR}/confusion_{int(datetime.timestamp(datetime.now()))}.png'
    if verbose:
        print(f'[INFO] Building and saving confusion matrix ({result_filepath})...')
        confusion = build_and_save_confusion_matrix(predictions, test_labels, (2,2), ['0', '1'], result_filepath)
    else:
        confusion = build_confusion_matrix(predictions, test_labels, (2,2))
    
    # PValue
    if verbose:
        model_stats(clf)

    # Accuracy
    precision = confusion[1,1] / (confusion[1,1] + confusion[0,1])
    accuracy = (confusion[1,1] + confusion[0,0]) / (np.sum(confusion))
    if verbose:
        print(f'[RESULT] Model accuracy is {accuracy}')
        print(f'[RESULT] Model precision is {precision}')
        print("\n\n")
    return precision, accuracy, clf

def run_normal(df, df_test, predictor_headers, result_headers, account_male_female = False):
    # Split dataset
    if df_test is None:
        train, test = split_dataset(df, TRAIN_TEST)
    else:
        train, test = df, df_test

    train, _, _, _, test_data, test_labels = dataset_train_test(train, test, predictor_headers, result_headers)
    
    # Perform the train & test operations
    _, _, clf = perform_train_test(train, test_data, test_labels, predictor_headers, result_headers, verbose=True)

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

####################################################################################
#################################### RUN CROSS #####################################
####################################################################################

def run_cross_validation_iteration(i, elements_per_bin, df, predictor_headers, result_headers, results):
    print('Running cross validation with bin number', i)
    # Dataset split
    test = df.iloc[i*elements_per_bin:(i+1)*elements_per_bin]
    train = df[~df.index.isin(list(test.index.values))]
    train, _, _, _, test_data, test_labels = dataset_train_test(train, test, predictor_headers, result_headers)
    # Perform operations
    precision, accuracy, _ = perform_train_test(train, test_data, test_labels, predictor_headers, result_headers, verbose=False)
    # Store the best out of all the runs
    results[i] = [accuracy, precision, train.copy(), test.copy()]

def run_cross_validation(df, cross_k, predictor_headers, result_headers):
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
            i, elements_per_bin, df, predictor_headers, result_headers, return_dict))
        jobs[i].start()
    # Join the jobs for the results
    for i in range(len(jobs)):
        jobs[i].join()
    # Calculate some metrics
    values = return_dict.values()
    accuracies = np.array([x[0] for x in values])
    precisions = np.array([x[1] for x in values])
    trains = np.array([x[2] for x in values])
    tests = np.array([x[3] for x in values])
    for i in range(accuracies.shape[0]):
        if BestSeparation.active and accuracies[i] > BestSeparation.getBestAccuracy():
            BestSeparation.setBestAccuracy(accuracies[i])
            BestSeparation.setDfs(trains[i], tests[i])
            BestSeparation.setCrossK(cross_k)
    print('---------------------------')
    print('---------------------------')
    print('---------------------------')
    print('Accuracy average -->', np.average(accuracies, axis=0),
          '\nstd -->', np.std(accuracies, axis=0))
    print('Precision average -->', np.average(precisions, axis=0),
          '\nstd -->', np.std(precisions, axis=0))
    return np.average(accuracies, axis=0), np.std(accuracies, axis=0), np.average(precisions, axis=0), np.std(precisions, axis=0)


####################################################################################
###################################### MODEL #######################################
####################################################################################

def get_model(data, predictor_headers, outcome_headers, disp = True):
    left_side = reduce(lambda a,b: a + ' + ' + b, outcome_headers)
    right_side = reduce(lambda a,b: a + ' + ' + b, predictor_headers)
    formula = left_side + " ~ " + right_side
    clf = smf.logit(formula=formula, data = data).fit(disp = disp)
    return clf

def use_model(clf, data):
    tmp_predictions = clf.predict(data)
    predictions = np.array(list(map(round, tmp_predictions)))
    return predictions

def run_logistic(df, cross_k = None, account_male_female = False, df_test = None):
    # Create result directory if not exists
    create_results_if_not_exist()

    # Shuffle dataset only if no test data is sent
    if df_test is None:
        df = shuffle_dataset(df)
    else:
        # Modify the headings so that they are properly named
        df_test = modify_dataset(df_test)
    # Modify the headings so that they are properly named
    df = modify_dataset(df)
    # Get headers to use later
    predictor_headers, result_headers = get_headers(account_male_female)

    # Normal execution
    if cross_k == None:
        run_normal(df, df_test, predictor_headers, result_headers, account_male_female=account_male_female)
    # Cross validation execution
    else:
        return run_cross_validation(df, cross_k, predictor_headers, result_headers)
    