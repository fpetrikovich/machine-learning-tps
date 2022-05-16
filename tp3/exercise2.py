from utils.confusion import get_accuracy, get_precision
from utils.plotting import plot_confusion_matrix
from config.configurations import Configuration
from datetime import datetime
from sklearn.svm import SVC
import os
import cv2
import numpy as np
import multiprocessing

# Test
COW = 'cow.jpg'
# Train
CIELO = 'cielo.jpg'
PASTO = 'pasto.jpg'
VACA = 'vaca.jpg'

# Base path is current folder
BASE_PATH = '.'

# Split percentage, 10%
BASE_SPLIT_PERCENTAGE = 0.1

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
    return cow

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
    train_indexes = np.array([x for x in range(dataset.shape[0]) if x < bin_index * elements_per_bin or x > (bin_index + 1) * elements_per_bin])
    test_indexes = np.array([x for x in range(dataset.shape[0]) if x >= bin_index * elements_per_bin and x <= (bin_index + 1) * elements_per_bin])
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
    svc = SVC(kernel=svm_kernel, C=svm_c)
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
    test_diff = np.abs(test_predictions - test_labels)
    error_abs = test_diff[test_diff > 0].shape[0]
    accuracy = 1 - (error_abs / test_predictions.shape[0])
    return test_predictions, error_abs, accuracy


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
            if predictions[pixel_number] == 1:
                test_image[row, col] = np.array([0, 0, 255])
            # Pasto
            elif predictions[pixel_number] == 2:
                test_image[row, col] = np.array([0, 255, 0])
            # Vaca
            elif predictions[pixel_number] == 3:
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
    train_dataset, train_labels, test_dataset, test_labels = split_dataset_bins(dataset, labels, i, elements_per_bin)
    # Perform the training & classification
    clf, _ = create_and_train_model(train_dataset, train_labels, svm_kernel, svm_c)
    _, error_abs, accuracy = predict_with_model(clf, test_dataset, test_labels)
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
        run_cross_validation_iteration(i, elements_per_bin, dataset, labels, svm_kernel, svm_c, return_dict)
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
        mode (str, optional): Execution mode, can be 'dataset' for analysis of the dataset and division or 'solve' for solving the excercise. Is used in combination of the cross_k paratemer for cross validation. Defaults to 'dataset'.
    """
    # Load images
    if Configuration.isVeryVerbose():
        print('[INFO] Loading images -', datetime.now())
    train_cielo, train_pasto, train_vaca = load_train_images(pic_folder_path)
    test_cow = load_test_images(pic_folder_path)
    # Build the dataset and shuffle it
    if Configuration.isVeryVerbose():
        print('[INFO] Building dataset -', datetime.now())
    dataset, labels = build_image_dataset(
        [train_cielo, train_pasto, train_vaca], [1, 2, 3])
    dataset, labels = shuffle_dataset(dataset, labels)
    if mode == 'dataset':
        if cross_k == None:
            a = 2
        else:
            run_cross_validation(dataset, labels, cross_k, svm_kernel, svm_c)
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
            print(1 - (test_diff[test_diff > 0].shape[0] / test_predictions.shape[0]))
            # Running the real test
            if Configuration.isVeryVerbose():
                print('[INFO] Running test -', datetime.now())
            run_test(clf, np.copy(test_cow))
        else:
            a = 2
