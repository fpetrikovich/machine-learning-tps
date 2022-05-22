from config.configurations import Configuration
from datetime import datetime
from sklearn.svm import SVC
import os
import cv2
import numpy as np
import multiprocessing
from utils.plotting import save_confusion_matrix
import pandas as pd

# Test
COW = 'cow.jpg'
COW2 = 'test--cow-2.jpeg'
COW3 = 'test--cow-3.jpeg'
COW4 = 'test--cow-4.jpeg'
# Train
CIELO = 'cielo.jpg'
PASTO = 'pasto.jpg'
VACA = 'vaca.jpg'

# Base path is current folder
BASE_PATH = '.'

# Split percentage, 10%
BASE_SPLIT_PERCENTAGE = 0.1

# Test parameters
# C_VALUES = np.array([0.001, 0.01, 0.1])
# KERNEL_VALUES = ['rbf']
# GAMMA_VALUES = np.array(['scale', 0.0001])
# DEGREE_VALUES = np.array([3])
C_VALUES = np.array([0.001, 0.01, 0.1, 1, 10])
KERNEL_VALUES = ['linear', 'poly', 'rbf', 'sigmoid']
GAMMA_VALUES = np.array(['scale', 0.0001, 0.01, 1, 10])
DEGREE_VALUES = np.array([2, 3, 4])

####################################################################################
################################# IMAGE OPERATIONS #################################
####################################################################################


def show_image(img, title='Image'):
    # Convert image back to BGR to display
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # Show it
    cv2.imshow(title, img)
    # Waits for user to press any key
    cv2.waitKey(0)
    # Closing all open windows
    cv2.destroyAllWindows()

####################################################################################
################################## IMAGE LOADING ###################################
####################################################################################


def load_image(path):
    # Default flag is color
    img = cv2.imread(path)
    # Convert to RGB color
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_train_images(pic_folder_path):
    cielo, pasto, vaca = load_image(os.path.join(BASE_PATH, pic_folder_path, CIELO)), load_image(
        os.path.join(BASE_PATH, pic_folder_path, PASTO)), load_image(os.path.join(BASE_PATH, pic_folder_path, VACA))
    return cielo, pasto, vaca


def load_test_images(pic_folder_path):
    cow = load_image(os.path.join(BASE_PATH, pic_folder_path, COW))
    cow2 = load_image(os.path.join(BASE_PATH, pic_folder_path, COW2))
    cow3 = load_image(os.path.join(BASE_PATH, pic_folder_path, COW3))
    cow4 = load_image(os.path.join(BASE_PATH, pic_folder_path, COW4))
    return [cow, cow2, cow3, cow4]

####################################################################################
#################################### ANALYSIS ######################################
####################################################################################


def analyze_image(img):
    print('Average color -->', np.mean(img, axis=(0, 1)))
    print('St. dev color -->', np.std(img, axis=(0, 1)))


def perform_image_analysis(cielo, pasto, vaca):
    print('---------------------------')
    print('IMAGE ANALYSIS')
    print('CIELO')
    analyze_image(cielo)
    print('PASTO')
    analyze_image(pasto)
    print('VACA')
    analyze_image(vaca)
    print('---------------------------')


def build_confusion_matrix(predictions, test_labels):
    confusion = np.zeros((3, 3))
    for i in range(test_labels.shape[0]):
        confusion[test_labels[i], predictions[i]] += 1
    return confusion

def build_and_save_confusion_matrix(predictions, test_labels, svm_kernel, svm_c):
    confusion = build_confusion_matrix(predictions, test_labels)
    save_confusion_matrix(confusion, ['CIELO', 'PASTO', 'VACA'],
                          f'./results/{svm_kernel}_{svm_c}_{int(datetime.timestamp(datetime.now()))}.png')

####################################################################################
############################### DATASET OPERATIONS #################################
####################################################################################


def map_pixels_to_datapoint(img, _class):
    # Outputs an np.array like [[np.array(pixel), class]]
    pixels, labels = [], []
    rows, cols = img.shape[0], img.shape[1]
    for row in range(rows):
        for col in range(cols):
            pixels.append(img[row, col, :])
            labels.append(_class)
    return np.array(pixels), np.array(labels)


def build_image_dataset(images, classes):
    pixels, labels = np.array([]), np.array([])
    for i in range(len(images)):
        _pixels, _labels = map_pixels_to_datapoint(images[i], classes[i])
        if i == 0:
            pixels = _pixels
            labels = _labels
        else:
            pixels = np.concatenate((pixels, _pixels))
            labels = np.concatenate((labels, _labels))
    return pixels, labels


def shuffle_dataset(dataset, labels):
    # As per https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    indexes = np.random.permutation(dataset.shape[0])
    return dataset[indexes], labels[indexes]

# 132k datapoints in the dataset


def split_dataset(dataset, labels, percentage):
    index_limit = int(dataset.shape[0] * percentage)
    train_dataset, test_dataset = dataset[index_limit:], dataset[:index_limit]
    train_labels, test_labels = labels[index_limit:], labels[:index_limit]
    return train_dataset, train_labels, test_dataset, test_labels


def split_dataset_bins(dataset, labels, bin_index, elements_per_bin):
    train_indexes = np.array([x for x in range(dataset.shape[0]) if x <
                             bin_index * elements_per_bin or x > (bin_index + 1) * elements_per_bin])
    test_indexes = np.array([x for x in range(dataset.shape[0]) if x >=
                            bin_index * elements_per_bin and x <= (bin_index + 1) * elements_per_bin])
    train_dataset, test_dataset = dataset[train_indexes], dataset[test_indexes]
    train_labels, test_labels = labels[train_indexes], labels[test_indexes]
    return train_dataset, train_labels, test_dataset, test_labels

####################################################################################
##################################### TRAINING #####################################
####################################################################################


def create_and_train_model(dataset, labels, svm_kernel, svm_c):
    # Apply SVM
    if Configuration.isVeryVerbose():
        print('[INFO] Creating SVM model -', datetime.now())
    svc = SVC(kernel=svm_kernel, C=svm_c, cache_size=2000)
    clf = svc.fit(dataset, labels)
    return clf, svc


def predict_with_model(clf, test_dataset, test_labels):
    # Run predictions
    if Configuration.isVeryVerbose():
        print('[INFO] Making predictions -', datetime.now())
    test_predictions = clf.predict(test_dataset)
    # Calculate metrics
    if Configuration.isVeryVerbose():
        print('[INFO] Computing metrics -', datetime.now())
    hits = np.array([0.0] * 3)
    for i in range(test_labels.shape[0]):
        if test_labels[i] == test_predictions[i]:
            hits[test_labels[i]] += 1
    # Iterate the classes
    for i in range(0, 3):
        hits[i] = float(hits[i]) / \
            float(test_labels[test_labels == i].shape[0])
    test_diff = np.abs(test_predictions - test_labels)
    error_abs = test_diff[test_diff > 0].shape[0]
    return test_predictions, error_abs, hits


####################################################################################
#################################### PREDICTIONS ###################################
####################################################################################


def run_test(classifier, test_image):
    if Configuration.isVeryVerbose():
        print('[INFO] Building test dataset -', datetime.now())
    dataset, _ = build_image_dataset([test_image], [0])
    if Configuration.isVeryVerbose():
        print('[INFO] Making test predictions -', datetime.now())
    predictions = classifier.predict(dataset)
    # Iterate image and apply predictions
    if Configuration.isVeryVerbose():
        print('[INFO] Transforming predictions -', datetime.now())
    rows, cols, pixel_number = test_image.shape[0], test_image.shape[1], 0
    for row in range(rows):
        for col in range(cols):
            # Cielo
            if predictions[pixel_number] == 0:
                test_image[row, col] = np.array([0, 0, 255])
            # Pasto
            elif predictions[pixel_number] == 1:
                test_image[row, col] = np.array([0, 255, 0])
            # Vaca
            elif predictions[pixel_number] == 2:
                test_image[row, col] = np.array([255, 0, 0])
            pixel_number += 1
    if Configuration.isVeryVerbose():
        print('[INFO] Showing image -', datetime.now())
    show_image(test_image)

####################################################################################
################################# CROSS VALIDATION #################################
####################################################################################


def run_cross_validation_iteration(i, elements_per_bin, dataset, labels, svm_kernel, svm_c, results):
    print('Running cross validation with bin number', i, datetime.now())
    # Perform the division
    train_dataset, train_labels, test_dataset, test_labels = split_dataset_bins(
        dataset, labels, i, elements_per_bin)
    # Perform the training & classification
    clf, _ = create_and_train_model(
        train_dataset, train_labels, svm_kernel, svm_c)
    predictions, error_abs, accuracy = predict_with_model(
        clf, test_dataset, test_labels)
    # Build the confusion matrix and store it
    build_and_save_confusion_matrix(predictions, test_labels, svm_kernel, svm_c)
    results[i] = [error_abs, accuracy]


def run_cross_validation(dataset, labels, cross_k, svm_kernel, svm_c):
    # Calculate number of elements per bin
    elements_per_bin = int(dataset.shape[0]/cross_k)
    print("Running cross validation using", cross_k,
          "bins with", elements_per_bin, "elements per bin", datetime.now())
    # Iterate and run method
    manager = multiprocessing.Manager()
    # Need this dictionary due to non-shared memory issues
    return_dict = manager.dict()
    jobs = [0] * cross_k
    # Create multiple jobs, SVC does not really like multithreadding
    for i in range(cross_k):
        run_cross_validation_iteration(
            i, elements_per_bin, dataset, labels, svm_kernel, svm_c, return_dict)
        # jobs[i] = multiprocessing.Process(target=run_cross_validation_iteration, args=(
        #     i, elements_per_bin, dataset, labels, svm_kernel, svm_c, return_dict))
        # jobs[i].start()
        # jobs[i].join()
    # Join the jobs for the results
    # for i in range(len(jobs)):
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

def run_single_svm(svc, train_dataset, train_labels, test_dataset, test_labels):
    clf = svc.fit(train_dataset, train_labels)
    if Configuration.isVeryVerbose():
        print('[INFO] Making predictions -', datetime.now())
    test_predictions = clf.predict(test_dataset)
    # Calculate metrics
    if Configuration.isVeryVerbose():
        print('[INFO] Computing metrics -', datetime.now())
    test_diff = np.abs(test_predictions - test_labels)
    error_abs = test_diff[test_diff > 0].shape[0]
    accuracy = 1 - (error_abs / test_predictions.shape[0])
    confusion = build_confusion_matrix(test_predictions, test_labels)
    return error_abs, accuracy, confusion, clf

def store_svm_data(errors, accuracies, uid):
    # Create DFs
    error_data = pd.DataFrame(data=errors)
    accuracy_data = pd.DataFrame(data=accuracies)
    # Store the data
    error_data.to_excel(f'./results/errors_{uid}.xlsx')
    accuracy_data.to_excel(f'./results/accuracies_{uid}.xlsx')

def run_multiple_svm(train_dataset, train_labels, test_dataset, test_labels):
    # Storing results
    uid = int(datetime.timestamp(datetime.now()))
    best_error, best_clf, best_config = 10000000, None, None
    # Data to convert to dataframe
    error_data = {}
    accuracy_data = {}
    # Initial data
    error_data['c'] = C_VALUES
    accuracy_data['c'] = C_VALUES
    for kernel in KERNEL_VALUES:
        if kernel == 'rbf':
            for g in GAMMA_VALUES:
                error_points = []
                accuracy_points = []
                for c in C_VALUES:
                    if Configuration.isVeryVerbose():
                        print(f'[INFO] Creating SVM model {kernel} c={c} g={g}', datetime.now())
                    svc = SVC(kernel=kernel, C=c, gamma=float(g) if g != 'scale' else g, cache_size=2000)
                    error_abs, accuracy, confusion, clf = run_single_svm(svc, train_dataset, train_labels, test_dataset, test_labels)
                    save_confusion_matrix(confusion, ['CIELO', 'PASTO', 'VACA'],
                            f'./results/{kernel}_{c}_{g}_{uid}.png')
                    error_points.append(error_abs)
                    accuracy_points.append(accuracy)
                    # Compare to get best configuration
                    if error_abs < best_error:
                        best_clf = clf
                        best_config = f'{kernel}_{c}_{g}'
                        best_error = error_abs
                error_data[f'{kernel}_{g}'] = error_points
                accuracy_data[f'{kernel}_{g}'] = accuracy_points
                store_svm_data(error_data, accuracy_data, uid)
        elif kernel == 'poly':
            for d in DEGREE_VALUES:
                error_points = []
                accuracy_points = []
                for c in C_VALUES:
                    if Configuration.isVeryVerbose():
                        print(f'[INFO] Creating SVM model {kernel} c={c} d={d}', datetime.now())
                    svc = SVC(kernel=kernel, C=c, degree=d, cache_size=2000)
                    error_abs, accuracy, confusion, clf = run_single_svm(svc, train_dataset, train_labels, test_dataset, test_labels)
                    save_confusion_matrix(confusion, ['CIELO', 'PASTO', 'VACA'],
                            f'./results/{kernel}_{c}_{d}_{uid}.png')
                    error_points.append(error_abs)
                    accuracy_points.append(accuracy)
                    # Compare to get best configuration
                    if error_abs < best_error:
                        best_clf = clf
                        best_config = f'{kernel}_{c}_{d}'
                        best_error = error_abs
                error_data[f'{kernel}_{d}'] = error_points
                accuracy_data[f'{kernel}_{d}'] = accuracy_points
                store_svm_data(error_data, accuracy_data, uid)
        else:
            error_points = []
            accuracy_points = []
            for c in C_VALUES:
                if Configuration.isVeryVerbose():
                    print(f'[INFO] Creating SVM model {kernel} c={c}', datetime.now())
                svc = SVC(kernel=kernel, C=c, cache_size=2000)
                error_abs, accuracy, confusion, clf = run_single_svm(svc, train_dataset, train_labels, test_dataset, test_labels)
                save_confusion_matrix(confusion, ['CIELO', 'PASTO', 'VACA'],
                          f'./results/{kernel}_{c}_{uid}.png')
                error_points.append(error_abs)
                accuracy_points.append(accuracy)
                # Compare to get best configuration
                if error_abs < best_error:
                    best_clf = clf
                    best_config = f'{kernel}_{c}'
                    best_error = error_abs
            error_data[f'{kernel}'] = error_points
            accuracy_data[f'{kernel}'] = accuracy_points
            store_svm_data(error_data, accuracy_data, uid)
    store_svm_data(error_data, accuracy_data, uid)
    return best_clf, best_config


####################################################################################
#################################### ENTRYPOINT ####################################
####################################################################################


def run_exercise_2(pic_folder_path, svm_c=1, svm_kernel='linear', cross_k=None, mode='dataset'):
    """_summary_

    Args:
        pic_folder_path (_type_): _description_
        svm_c (int, optional): _description_. Defaults to 1.
        svm_kernel (str, optional): _description_. Defaults to 'linear'.
        cross_k (_type_, optional): _description_. Defaults to None.
        mode (str, optional): Execution mode, can be 'dataset' for analysis of the dataset and division or 'solve' for solving the excercise. The 'analyze-image' mode is used to gather analytics over images. The 'analyze' mode is used to test SVM. Is used in combination of the cross_k paratemer for cross validation. Defaults to 'dataset'.
    """
    # Load images
    if Configuration.isVeryVerbose():
        print('[INFO] Loading images -', datetime.now())
    train_cielo, train_pasto, train_vaca = load_train_images(pic_folder_path)
    test_cows = load_test_images(pic_folder_path)
    if mode == 'analyze-image':
        perform_image_analysis(train_cielo, train_pasto, train_vaca)
    else:
        # Build the dataset and shuffle it
        if Configuration.isVeryVerbose():
            print('[INFO] Building dataset -', datetime.now())
        dataset, labels = build_image_dataset(
            [train_cielo, train_pasto, train_vaca], [0, 1, 2])
        dataset, labels = shuffle_dataset(dataset, labels)
        if mode == 'dataset':
            if cross_k == None:
                print("[ERROR] This parameter combination isn't supported")
            else:
                run_cross_validation(
                    dataset, labels, cross_k, svm_kernel, svm_c)
        elif mode == 'solve':
            if cross_k == None:
                # Run the full prediction flow
                # Split the dataset in train/test
                if Configuration.isVeryVerbose():
                    print('[INFO] Splitting dataset -', datetime.now())
                train_dataset, train_labels, test_dataset, test_labels = split_dataset(
                    dataset, labels, percentage=BASE_SPLIT_PERCENTAGE)
                # Apply SVM
                if Configuration.isVeryVerbose():
                    print('[INFO] Creating SVM model -', datetime.now())
                svc = SVC(kernel=svm_kernel, C=svm_c)
                clf = svc.fit(train_dataset, train_labels)
                if Configuration.isVeryVerbose():
                    print('[INFO] Making predictions -', datetime.now())
                test_predictions = clf.predict(test_dataset)
                # Calculate metrics
                if Configuration.isVeryVerbose():
                    print('[INFO] Computing metrics -', datetime.now())
                test_diff = np.abs(test_predictions - test_labels)
                print(
                    1 - (test_diff[test_diff > 0].shape[0] / test_predictions.shape[0]))
                # Running the real test
                for test_cow in test_cows:
                    if Configuration.isVeryVerbose():
                        print('[INFO] Running test - COW -', datetime.now())
                    run_test(clf, np.copy(test_cow))
            else:
                print("[ERROR] This parameter combination isn't supported")
        elif mode == 'analyze':
            # Split the dataset in train/test
            if Configuration.isVeryVerbose():
                print('[INFO] Splitting dataset -', datetime.now())
            # Split dataset in 60-40, then split 40 in 20-20.
            # Use 60 to train, 20 to test, 20 to test best SVM
            train_dataset, train_labels, test_dataset, test_labels = split_dataset(
                dataset, labels, percentage=.6)
            test_all_dataset, test_all_labels, test_best_dataset, test_best_labels = split_dataset(
                test_dataset, test_labels, percentage=0.5)
            clf, config = run_multiple_svm(train_dataset, train_labels, test_all_dataset, test_all_labels)
            print(f'[RESULT] Best SVM configuration is {config} -', datetime.now())
            test_predictions, error_abs, hits = predict_with_model(clf, test_best_dataset, test_best_labels)
            print(f'[RESULT] Error for best SVM is {error_abs} -', datetime.now())
            confusion = build_confusion_matrix(test_predictions, test_best_labels)
            save_confusion_matrix(confusion, ['CIELO', 'PASTO', 'VACA'],
                            f'./results/best_configuration.png')
