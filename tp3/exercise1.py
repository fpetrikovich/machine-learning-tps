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

def generate_points(amount, margin, misclassifications, a,b,c):
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
    c = y[x.index(0)]

    v = [a,b,c]
    normalized_v = v/np.linalg.norm(v)
    return normalized_v[0], normalized_v[1], normalized_v[2]

def run_exercise_1():
    # Usamos la recta "y = x"
    x = np.linspace(0, 5, 10)
    y = x
    a,b,c = get_hyperplane_vector(x,y)
    points = generate_points(25, 0.5, 0, a,b,c)
    perceptron = SimplePerceptron()
    w = perceptron.algorithm(points) # w = [b0, b1, b2] = [c, a, b]
    y_perceptron = -w[1]/w[2] * x - w[0]/w[2]
    # TODO: Buscar hiperplano óptimo
    # TODO: Usar SVM para clasificar
    plt.scatter(points[:,0], points[:,1], c=points[:,2])
    plt.plot(x, y, color='blue')
    plt.plot(x, y_perceptron, color='red')
    plt.xlim(-0.1, 5.1)
    plt.ylim(-0.1, 5.1)
    plt.show()
