from fileHandling import read_csv, df_to_numpy, print_entire_df
from utils.plotting import *
from kmedias import KMeans
from hierarchy_new import Hierarchy
import kohonen.kohonen as kohonen
import numpy as np
from config.constants import Headers, Similarity_Methods
from utils.confusion import build_confusion_matrix
from sklearn.model_selection import train_test_split

def standard_to_normal(name, matrix, min, max):
    return matrix * (max[name]-min[name]) + min[name]

def map_classifications(df):
    vals = df.unique()
    mapping = {}
    for i in range(len(vals)):
        mapping[vals[i]] = i
    return mapping, vals

def run_KMeans(file):
    df = read_csv(file)
    df = df[(df[Headers.GENRES.value] == 'Drama') | (df[Headers.GENRES.value] == 'Comedy') | (df[Headers.GENRES.value] == 'Action')]
    class_mapping, class_values = map_classifications(df[Headers.GENRES.value])
    classifications = df.apply(lambda row : class_mapping[row[Headers.GENRES.value]], axis = 1)
    # props = [Headers.POPULARITY.value, Headers.REVENUE.value, Headers.BUDGET.value]
    props = [Headers.BUDGET.value, Headers.POPULARITY.value, Headers.REVENUE.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value]
    df = df[props]
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

        for cluster_amount in [3]:
            kmeans = KMeans(X_train, y_train, cluster_amount)
            classes = kmeans.apply()
            stereotypes = kmeans.get_stereotypes(extremes)
            for i in range(len(stereotypes)):
                print("Stereotype for cluster #", i, ":", stereotypes[i])

            expected = []
            predictions = kmeans.predict(X_test)
            for i in range(len(y_test)):
                expected.append(y_test[i])
            if(len(X_test) > 0):
                confusion = build_confusion_matrix(np.asarray(predictions), np.asarray(expected), [len(class_mapping),len(class_mapping)])
                accuracy = 0
                for i in range(len(class_mapping)):
                    accuracy += confusion[i][i]
                accuracy = accuracy / np.sum(confusion)
                if cluster_amount not in results:
                    results[cluster_amount] = [accuracy]
                else:
                    results[cluster_amount].append(accuracy)
                save_confusion_matrix(confusion, class_values, "results/kmeans-matrix-"+str(cluster_amount))
            plot_save_hierarchy(classes, X_train, X_test, y_test, props, "results/kmeans-plot-"+str(cluster_amount))
    plot_accuracy_evolution(results)

def run_kohonen(file, k, iterations):
    df = read_csv(file)
    df = df[[Headers.BUDGET.value, Headers.POPULARITY.value, Headers.REVENUE.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value]]
    dfmin, dfmax = df.min(), df.max()
    df_standard = (df-dfmin)/(dfmax-dfmin)
    df_standard = df_standard[[Headers.BUDGET.value, Headers.POPULARITY.value, Headers.REVENUE.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value]]
    df_standard = df_to_numpy(df_standard)

    config = Config(k, iterations)
    kohonen.apply(config, df_standard, [Headers.BUDGET.value, Headers.POPULARITY.value, Headers.REVENUE.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value], lambda name, matrix: standard_to_normal(name, matrix, dfmin, dfmax))

def run_hierarchy(file):
    df = read_csv(file)
    df = df[(df[Headers.GENRES.value] == 'Drama') | (df[Headers.GENRES.value] == 'Comedy') | (df[Headers.GENRES.value] == 'Action')]
    class_mapping, class_values = map_classifications(df[Headers.GENRES.value])
    classifications = df.apply(lambda row : class_mapping[row[Headers.GENRES.value]], axis = 1)
    # props = [Headers.POPULARITY.value, Headers.REVENUE.value, Headers.BUDGET.value]
    props = [Headers.BUDGET.value, Headers.POPULARITY.value, Headers.REVENUE.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value]
    df = df[props]
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

        #X_train = np.concatenate([X_train, X_test])
        #y_train = np.concatenate([y_train, y_test])
        #X_test = []
        #y_test = []

        hierarchy = Hierarchy(X_train, y_train, Similarity_Methods.CENTROID)
        for cluster_amount in [3, 2]:
            classes = hierarchy.run(cluster_amount)
            stereotypes = hierarchy.get_stereotypes(extremes)
            for i in range(len(stereotypes)):
                print("Stereotype for cluster #", i, ":", stereotypes[i])

            expected = []
            predictions = hierarchy.predict(X_test)
            for i in range(len(y_test)):
                expected.append(int(y_test[i]))
            if(len(X_test) > 0):
                confusion = build_confusion_matrix(np.asarray(predictions), np.asarray(expected), [len(class_mapping),len(class_mapping)])
                accuracy = 0
                for i in range(len(class_mapping)):
                    accuracy += confusion[i][i]
                if cluster_amount not in results:
                    results[cluster_amount] = [accuracy]
                else:
                    results[cluster_amount].append(accuracy)
                save_confusion_matrix(confusion, class_values, "results/hierarchy-matrix-"+str(cluster_amount))
            plot_save_hierarchy(classes, X_train, X_test, y_test, props, "results/hierarchy-plot-"+str(cluster_amount))
    plot_accuracy_evolution(results)

def plot_2d_example():
    a = 4

class Config:
    def __init__(self, k, iterations):
        self.k = k
        self.iterations = iterations
