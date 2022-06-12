# External
from sklearn.linear_model import LogisticRegression
from datetime import datetime
import os
import matplotlib.pyplot as plt

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
###################################### STATS #######################################
####################################################################################

def variable_importance(clf, cols):
    importance = clf.coef_[0]

    for i,j in enumerate(importance):
        print('Feature: %s, Score: %.5f' % (cols[i],j))

    plt.bar([X for X in range(len(importance))], importance)
    plt.show()

####################################################################################
###################################### MODEL #######################################
####################################################################################

def get_model(data, labels):
    clf = LogisticRegression(random_state=0).fit(data, labels)
    return clf

def run_logistic(df, cross_k = None, account_male_female = False):
    # Create result directory if not exists
    create_results_if_not_exist()
    # Shuffle dataset
    df = shuffle_dataset(df)
    # Split dataset
    train, test = split_dataset(df, TRAIN_TEST)
    # Get data and labels division
    train_data, train_labels = split_dataset_data_labels(train, [Headers.AGE.value, Headers.CAD_DUR.value, Headers.CHOLESTEROL.value], [Headers.TVDLM.value])
    test_data, test_labels = split_dataset_data_labels(test, [Headers.AGE.value, Headers.CAD_DUR.value, Headers.CHOLESTEROL.value], [Headers.TVDLM.value])
    # Get the logistics model
    # We have to ravel in order to get a proper array if labels
    clf = get_model(train_data, train_labels.values.ravel())
    # Making predictions
    predictions = clf.predict(test_data)
    # Confusion
    result_filepath = f'{RESULTS_DIR}/confusion_{int(datetime.timestamp(datetime.now()))}.png'
    print(f'[INFO] Building and saving confusion matrix ({result_filepath})...')
    build_and_save_confusion_matrix(predictions, test_labels.values.ravel(), (2,2), ['0', '1'], result_filepath)
    # PValue
    variable_importance(clf, train_data.columns.to_list())