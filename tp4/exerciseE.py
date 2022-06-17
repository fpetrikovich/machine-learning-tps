from fileHandling import read_csv, df_to_numpy
from utils.plotting import plot_hierarchy, plot_confusion_matrix
from kmedias import KMeans
from hierarchy_new import Hierarchy
import numpy as np
from config.constants import Headers, Similarity_Methods
from utils.confusion import build_confusion_matrix
from sklearn.model_selection import train_test_split


def run_KMeans(file, k):
    df = read_csv(file)
    df = df[[Headers.AGE.value, Headers.CAD_DUR.value, Headers.CHOLESTEROL.value]]
    df_np = df_to_numpy(df)

    kmeans = KMeans(df_np, k)
    point, centroide = kmeans.apply()

def run_hierarchy(file):
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

    hierarchy = Hierarchy(X_train, y_train, Similarity_Methods.CENTROID)
    classes = hierarchy.run(2)
    predictions = []
    expected = []
    predictions = hierarchy.predict(X_test)
    for i in range(len(y_test)):
        expected.append(int(y_test[i][0]))
    confusion = build_confusion_matrix(np.asarray(predictions), np.asarray(expected), [2,2])
    plot_confusion_matrix(confusion, ["Healthy", "Ill"])
    plot_hierarchy(classes, X_train, X_test, y_test)

def plot_2d_example():
    a = 4
