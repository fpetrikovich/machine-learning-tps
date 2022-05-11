from utils.confusion import get_accuracy, get_precision
from utils.plotting import plot_confusion_matrix
from config.configurations import Configuration
from perceptron import SimplePerceptron
import matplotlib.pyplot as plt
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
        f = a*p[0] + b*p[1] + c
        # Si está dentro del margen, le buscamos otro valor
        while calculate_distance(p, a,b,c) < margin:
            p[0] = np.random.random()*5
            p[1] = np.random.random()*5
            f = a*p[0] + b*p[1] + c
        # Tercer columna es la clasificación
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

def run_exercise_1():
    # Usamos la recta "y = x"
    x = np.linspace(0, 5, 10)
    y = x
    a,b,c = get_hyperplane_vector(x,y)
    points = generate_points(25, 0.25, 0, a,b,c)
    perceptron = SimplePerceptron()

    w, margin = perceptron.algorithm(points) # w = [b0, b1, b2] = [c, a, b]
    y_perceptron = -w[1]/w[2] * x - w[0]/w[2]
    optimal_vector, optimal_margin, optimal_points, optimal_dist_y = perceptron.optimal_hiperplane(w, points, 4)
    y_optimal = -optimal_vector[1]/optimal_vector[2] * x - optimal_vector[0]/optimal_vector[2]

    print("Perceptron margin:", margin, "\nPerceptron Weights: ", w)
    print("Optimal margin:", optimal_margin, "\nOptimal Weights: ", optimal_vector)
    plt.scatter(points[:,0], points[:,1], c=points[:,2])
    #plt.plot(x, y, color='grey', linestyle='dotted', alpha=0.25)
    plt.plot(x, y_perceptron, color='red', label='Perceptron')
    plt.plot(x, y_optimal, color='green', label='Óptimo')
    plt.plot(x, y_optimal+optimal_dist_y/2, color='green', linestyle='dotted', alpha=0.5)
    plt.plot(x, y_optimal-optimal_dist_y/2, color='green', linestyle='dotted', alpha=0.5)
    plt.xlim(-0.1, 5.1)
    plt.ylim(-0.1, 5.1)
    plt.legend()
    plt.show()
    # TODO: Usar SVM para clasificar
