from utils.confusion import get_accuracy, get_precision
from utils.plotting import plot_confusion_matrix
from config.configurations import Configuration
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import datetime
import math
import random

def generate_points(amount, margin, misclassifications, a,b,c):
    points = np.random.random((amount,3)) * 5
    for p in points:
        # Si está dentro del margen, le buscamos otro valor
        while abs(p[1]-p[0]) < margin:
            p[1] = np.random.random()*5
        # Tercer columna es la clasificación
        if p[1] > p[0]:
            p[2] = 1
        else:
            p[2] = -1
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
    return a,b,c

def run_exercise_1():
    # Usamos la recta "y = x"
    x = np.linspace(0, 5, 10)
    y = x
    a,b,c = get_hyperplane_vector(x,y)

    points = generate_points(25, 0.5, 2, a,b,c)
    # TODO: Crear perceptron y clasificar
    # TODO: Buscar hiperplano óptimo
    # TODO: Usar SVM para clasificar
    plt.scatter(points[:,0], points[:,1], c=points[:,2])
    plt.plot(x, y, color='blue')
    plt.show()
