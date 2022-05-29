from graphing import graphR2Hiperplane
from utils.confusion import get_accuracy, get_precision
from utils.plotting import plot_confusion_matrix
from config.configurations import Configuration
from perceptron import SimplePerceptron
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import multiprocessing
import datetime
import math
import random

def generate_points(amount, margin, misclassifications, a,b,c, seed=None):
    if seed:
        np.random.seed(seed)
    points = np.random.random((amount,3)) * 5
    for p in points:
        # Si está dentro del margen, le buscamos otro valor
        while calculate_distance(p, a,b,c) < margin:
            p[0] = np.random.random()*5
            p[1] = np.random.random()*5
        # Tercer columna es la clasificación
        f = a*p[0] + b*p[1] + c
        p[2] = np.sign(f)
    # Ordenamos por distancia a la recta
    points = list(points)
    points.sort(key=lambda p: calculate_distance(p, a, b, c))
    # Intencionalmente clasificar mal aquellos más cercanos a la recta
    for i in range(misclassifications):
        points[i][2] *= -1
    return np.array(points)

def calculate_distance(p, a, b, c):
    # Line is 0 = ax + by + c
    return abs(a*p[0] + b*p[1] + c) / math.sqrt(a**2 + b**2)

def get_hyperplane_vector(x,y):
    x = list(x)
    a = (y[-1]-y[0])/(x[-1]-x[0])
    b = -1
    c = -a*x[-1] - b*y[-1]
    return [a,b,c]

def run_exercise_1(n=25, misclassifications=0, seed=None, iterations=5000, m=4, svm_C=1, printout=True):
    # Usamos la recta "y = x"
    x = np.linspace(0, 5, 10)
    y = x
    a,b,c = get_hyperplane_vector(x,y)
    points = generate_points(n, 0.05, misclassifications, a,b,c, seed)
    perceptron = SimplePerceptron(iterations=iterations)

    # Perceptron
    w, margin = perceptron.algorithm(points) # w = [b0, b1, b2] = [c, a, b]
    y_perceptron = -w[1]/w[2] * x - w[0]/w[2] # recta que dibujo el perceptron
    optimal_vector, optimal_margin, optimal_dist_y = perceptron.optimal_hiperplane(w, points, m)
    if printout:
        print("RECTA\t\tMargin:", perceptron.calculate_margin(points, a,b,c), "\tError: ", perceptron.test_classifier(points, [c,a,b]), "\tWeights: ", [c,a,b])
        print("PERCEPTRON\tMargin:", margin, "\tError: ", perceptron.test_classifier(points, w), "\tWeights: ", w)
    if optimal_vector:
        y_optimal = -optimal_vector[1]/optimal_vector[2] * x - optimal_vector[0]/optimal_vector[2]
        if printout:
            print("ÓPTIMO\t\tMargin:", optimal_margin, "\tError: ", perceptron.test_classifier(points, optimal_vector), "\tWeights: ", optimal_vector)

    # SVM
    svc = SVC(C=svm_C, kernel='linear')
    clf = svc.fit(points[:,0:2], points[:,2])
    w_svm = svc.coef_[0]
    w_svm = [svc.intercept_[0], w_svm[0], w_svm[1]]
    margin_svm = 1 / np.sqrt(np.sum(clf.coef_ ** 2))
    a_svm = (-w_svm[1] / w_svm[2])
    y_svm = a_svm*x - w_svm[0]/w_svm[2]
    y_svm_down = y_svm - np.sqrt(1 + a_svm**2) * margin_svm
    y_svm_up = y_svm + np.sqrt(1 + a_svm**2) * margin_svm
    if printout:
        print("SVM\t\tMargin:", perceptron.calculate_margin(points, w_svm[1], w_svm[2], w_svm[0]),
            "\tError: ", perceptron.test_classifier(points, w_svm), "\tWeights: ", w_svm)

    # Plot
    # plt.scatter(points[:,0], points[:,1], c=points[:,2])
    # plt.plot(x, y_perceptron, color='red', label='Perceptron')

    # plt.plot(x, y_svm, color='blue', label='SVM')
    # plt.plot(x, y_svm_down, color='blue', linestyle='dashed', alpha=0.5)
    # plt.plot(x, y_svm_up, color='blue', linestyle='dashed', alpha=0.5)
    # plt.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors='none', edgecolors='k')

    # if optimal_vector:
    #     plt.plot(x, y_optimal, color='green', label='Óptimo')
    #     plt.plot(x, y_optimal+optimal_dist_y/2, color='green', linestyle='dotted', alpha=0.5)
    #     plt.plot(x, y_optimal-optimal_dist_y/2, color='green', linestyle='dotted', alpha=0.5)
    # plt.xlim(-0.1, 5.1)
    # plt.ylim(-0.1, 5.1)
    # plt.legend()
    # if printout:
    #     plt.show()
    # plt.clf()

    graphR2Hiperplane(points, perceptron.getEpochHiperplanes())

    resp = [perceptron.calculate_margin(points, a,b,c), perceptron.test_classifier(points, [c,a,b]),
            margin, perceptron.test_classifier(points, w),
            perceptron.calculate_margin(points, w_svm[1], w_svm[2], w_svm[0]), perceptron.test_classifier(points, w_svm)]
    if optimal_vector:
        resp.append(optimal_margin)
        resp.append(perceptron.test_classifier(points, optimal_vector))
    return resp
