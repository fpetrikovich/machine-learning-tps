import numpy as np
from utils.plotting import save_confusion_matrix

def build_confusion_matrix(predictions, test_labels, target_shape):
    confusion = np.zeros(target_shape)
    for i in range(test_labels.shape[0]):
        confusion[test_labels[i], predictions[i]] += 1
    return confusion

def build_and_save_confusion_matrix(predictions, test_labels, target_shape, labels, output_filename):
    confusion = build_confusion_matrix(predictions, test_labels, target_shape)
    save_confusion_matrix(confusion, labels, output_filename)