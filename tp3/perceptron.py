import numpy as np
import random
import math
from datetime import datetime

class SimplePerceptron:

    def __init__(self, alpha=0.01, iterations=1000, adaptive=False):
        self.alpha = alpha
        self.initial_alpha = alpha
        self.iterations = iterations
        self.adaptive = adaptive
        self.iterativeHiperplanes = []

    def adjust_learning_rate(self, errors_so_far):
        if(len(errors_so_far) > 10):
            last_10_errors = errors_so_far[-10:]
            # indicates if error at index i is bigger than the next error
            booleans = []
            for i in range(len(last_10_errors) - 1):
                booleans.append(last_10_errors[i] > last_10_errors[i + 1])
            if all(booleans):
                # If all are bigger ==> speed up the learning rate
                self.alpha += 0.001
            elif not all(booleans):
                # If no all are bigger ==> slow down the learning rate
                self.alpha -= 0.01 * self.alpha

    def step(self, x): # funcion de activacion escalon
        if x > 0.0: return 1.0
        if x < 0.0: return -1.0
        else: return 0.0

    def get_sum(self, xi, weights):
        sumatoria = 0.0
        for i,w in zip(xi, weights):
            sumatoria += i * w
        return sumatoria

    def get_activation(self, sumatoria):
        return self.step(sumatoria)

    def is_valid_classifier(self, data, margin, a,b,c):
        for p in data:
            if (a*p[0] + b*p[1] + c)*p[2] < 0:
                return False
            if self.calculate_distance(p, a,b,c) < margin:
                return False
        return True

    def calculate_distance(self, p, a, b, c):
        return abs(a*p[0] + b*p[1] + c) / math.sqrt(a**2 + b**2)

    def calculate_margin(self, data, a,b,c):
        distances = []
        for p in data:
            distances.append(self.calculate_distance(p,a,b,c))
        return min(distances)

    def algorithm(self, input):
        data = []
        for p in input:
            # Adding value to each data set to calculate the bias w0
            data.append([1.0, p[0], p[1], p[2]])
        data = np.array(data)
        # Initialize the weights randomly
        weights = np.random.rand(len(data[0]) - 1, 1)
        # setting min w and error in case the loop is finished by iterations
        error_min = len(data) * 2
        error_this_epoch = 1
        w_min = weights
        error_per_epoch = []
        for epoch in range(self.iterations):
            if error_this_epoch > 0:
                # Fetch a random data point
                i = np.random.randint(0,len(input))
                p = data[i]
                # 0 if no error 2 or -2 if error in prediction
                error = p[-1] - self.make_prediction(p, weights)
                # Weight correction
                for j in range(len(weights)):
                    # Correct the perceptron weights if there was an error
                    weights[j] = weights[j] + (self.alpha * error * p[j])
                # Calculate the error in all the data set for these weights
                error_this_epoch = self.test_perceptron(data, weights)
                error_per_epoch.append(error_this_epoch)
                # Save the weights
                self.iterativeHiperplanes.append(np.copy(weights))
                # Cada 10 epocas, si el perceptron es adaptativo
                if self.adaptive and epoch % 10 == 0:
                    self.adjust_learning_rate(error_per_epoch)
                # Updatear el error minimo si es necesario
                if error_this_epoch < error_min:
                    error_min = error_this_epoch
                    w_min = weights
            else:
                # If error is 0, leave the for loop
                break
        w_min = w_min[:,0]
        plane_margin = self.calculate_margin(input, w_min[1], w_min[2], w_min[0])
        return w_min, plane_margin

    def optimal_hiperplane(self, weights, data, n=4):
        data = list(data)
        data.sort(key=lambda t: self.calculate_distance(t, weights[1], weights[2], weights[0]))
        class1 = []
        class2 = []
        # Tomar los n puntos mas cercanos al hiperplano de cada clase
        for p in data:
            if p[2] == 1 and len(class1) < n:
                class1.append(p)
            if p[2] == -1 and len(class2) < n:
                class2.append(p)

        best_margin = -1
        best_vector = None
        best_points = None
        best_dist_y = None

        # Dual class => de donde tomo 2 puntos
        # Single class => de donde tomo 1 punto
        for dual_class in [class1, class2]:
            single_class = class1
            if dual_class is single_class:
                single_class = class2
            
            # Para cada punto del single_class, tomo 2 del dual class
            for single_point in single_class:
                for i in range(len(dual_class)):
                    for j in range(i+1, len(dual_class)):
                        single_point = single_point
                        dual_p1 = dual_class[i]
                        dual_p2 = dual_class[j]
                        # Hiperplano generado por dos puntos:
                        a = (dual_p2[1]-dual_p1[1])/(dual_p2[0]-dual_p1[0])
                        b = -1
                        c = dual_p2[1] - a*dual_p2[0]
                        # Buscamos paralela que atraviese punto singular y buscamos la recta entre esas dos
                        dist_y = single_point[1] - (a*single_point[0]+c)
                        c += dist_y/2
                        # Si este hiperplano clasifica bien y es maximal, lo guardamos
                        dist_dual_1 = self.calculate_distance(dual_p1, a,b,c)
                        dist_dual_2 = self.calculate_distance(dual_p2, a,b,c)
                        dist_single = self.calculate_distance(single_point, a,b,c)
                        if self.is_valid_classifier(data, min([dist_dual_1, dist_dual_2, dist_single]), a,b,c):
                            plane_margin = self.calculate_margin(data, a,b,c)
                            if plane_margin > best_margin:
                                best_margin = plane_margin
                                best_vector = [c,a,b]
                                best_points = np.array([single_point, dual_p1, dual_p2])
                                best_dist_y = dist_y
        return best_vector, best_margin, best_dist_y

    def make_prediction(self, p, w):
        return self.get_activation(self.get_sum(p[:-1], w))

    def test_perceptron(self, train_data, w):
        error = 0.0
        for p in train_data:
            error += abs(p[-1] - self.make_prediction(p,w))
        return error

    def test_classifier(self, test_data, w):
        error = 0.0
        predictions = []
        for p in test_data:
            sumatoria = w[0] + w[1]*p[0] + w[2]*p[1]
            predictions.append([p[2], np.sign(sumatoria)])
            if np.sign(sumatoria) != np.sign(p[2]):
                error += 1
        return error/len(test_data)

    def getEpochHiperplanes(self):
        return self.iterativeHiperplanes