import numpy as np
from enum import Enum


class ConfusionType(Enum):
    TP = 'TP'
    TN = 'TN'
    FP = 'FP'
    FN = 'FN'


def get_data_from_matrix(confusion):
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    return {ConfusionType.TP: TP, ConfusionType.TN: TN, ConfusionType.FP: FP, ConfusionType.FN: FN}


def get_error(confusion):
    mistakes = 0
    for i in range(len(confusion[0])):
        for j in range(len(confusion[0])):
            if i != j:
                mistakes += confusion[i, j]
    return mistakes


def get_accuracy(confusion):
    data = get_data_from_matrix(confusion)
    TP = data[ConfusionType.TP]
    FP = data[ConfusionType.FP]
    TN = data[ConfusionType.TN]
    FN = data[ConfusionType.FN]
    return (TP+TN)/(TP+TN+FP+FN)


def get_precision(confusion):
    data = get_data_from_matrix(confusion)
    TP = data[ConfusionType.TP]
    FP = data[ConfusionType.FP]
    return TP / (TP+FP)
