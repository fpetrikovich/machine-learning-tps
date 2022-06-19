from fileHandling import read_csv, df_to_numpy
from utils.plotting import *
from kmedias import KMeans
from hierarchy_new import Hierarchy
import kohonen.kohonen as kohonen
import numpy as np
from config.constants import Headers, Similarity_Methods
from utils.confusion import build_confusion_matrix
from sklearn.model_selection import train_test_split


def run_KMeans(file, k):
    df = read_csv(file)
    classifications = df[[Headers.SIGDZ.value]]
    df = df[[Headers.AGE.value, Headers.CAD_DUR.value, Headers.CHOLESTEROL.value]]
    extremes = [df_to_numpy(df.min()), df_to_numpy(df.max())]
    df = (df-df.min())/(df.max()-df.min())

    results = {}
    for attempt in range(5):
        print("Attempt #", attempt)
        X_train, X_test, y_train, y_test = train_test_split(df, classifications, test_size=0.1)
        X_train = df_to_numpy(X_train)
        X_test = df_to_numpy(X_test)
        y_train = df_to_numpy(y_train)
        y_test = df_to_numpy(y_test)

        for cluster_amount in [10, 5, 3, 2]:
            kmeans = KMeans(X_train, y_train, cluster_amount)
            classes = kmeans.apply()
            stereotypes = kmeans.get_stereotypes(extremes)
            for i in range(len(stereotypes)):
                print("Stereotype for cluster #", i, ":", stereotypes[i])

            expected = []
            predictions = kmeans.predict(X_test)
            for i in range(len(y_test)):
                expected.append(int(y_test[i][0]))
            confusion = build_confusion_matrix(np.asarray(predictions), np.asarray(expected), [2,2])
            accuracy = (confusion[0][0]+confusion[1][1]) / np.sum(confusion)
            if cluster_amount not in results:
                results[cluster_amount] = [accuracy]
            else:
                results[cluster_amount].append(accuracy)
            save_confusion_matrix(confusion, ["Healthy", "Ill"], "results/kmeans-matrix-"+str(cluster_amount))
            plot_save_hierarchy(classes, X_train, X_test, y_test, "results/kmeans-plot-"+str(cluster_amount))
    plot_accuracy_evolution(results)

def run_kohonen(file, k, iterations):
    df = read_csv(file)
    df = (df-df.min())/(df.max()-df.min())

    # Train-Test split
    classifications = df[[Headers.SIGDZ.value]]
    df = df[[Headers.AGE.value, Headers.CAD_DUR.value, Headers.CHOLESTEROL.value]]
    X_train, X_test, y_train, y_test = train_test_split(df, classifications, test_size=0.25)
    X_train = df_to_numpy(X_train)
    X_test = df_to_numpy(X_test)
    y_train = df_to_numpy(y_train)
    y_test = df_to_numpy(y_test)

    config = Config(k, iterations)
    kohonen.apply(config, X_train, y_train)

def run_hierarchy(file):
    df = read_csv(file)
    classifications = df[[Headers.SIGDZ.value]]
    df = df[[Headers.AGE.value, Headers.CAD_DUR.value, Headers.CHOLESTEROL.value]]
    extremes = [df_to_numpy(df.min()), df_to_numpy(df.max())]
    df = (df-df.min())/(df.max()-df.min())

    results = {}
    for attempt in range(3):
        print("Attempt #", attempt)
        X_train, X_test, y_train, y_test = train_test_split(df, classifications, test_size=0.1)
        X_train = df_to_numpy(X_train)
        X_test = df_to_numpy(X_test)
        y_train = df_to_numpy(y_train)
        y_test = df_to_numpy(y_test)

        hierarchy = Hierarchy(X_train, y_train, Similarity_Methods.CENTROID)
        for cluster_amount in [10, 5, 3, 2]:
            classes = hierarchy.run(cluster_amount)
            stereotypes = hierarchy.get_stereotypes(extremes)
            for i in range(len(stereotypes)):
                print("Stereotype for cluster #", i, ":", stereotypes[i])

            expected = []
            predictions = hierarchy.predict(X_test)
            for i in range(len(y_test)):
                expected.append(int(y_test[i][0]))
            confusion = build_confusion_matrix(np.asarray(predictions), np.asarray(expected), [2,2])
            accuracy = (confusion[0][0]+confusion[1][1]) / np.sum(confusion)
            if cluster_amount not in results:
                results[cluster_amount] = [accuracy]
            else:
                results[cluster_amount].append(accuracy)
            save_confusion_matrix(confusion, ["Healthy", "Ill"], "results/hierarchy-matrix-"+str(cluster_amount))
            plot_save_hierarchy(classes, X_train, X_test, y_test, "results/hierarchy-plot-"+str(cluster_amount))
    plot_accuracy_evolution(results)

def plot_2d_example():
    a = 4

class Config:
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations
