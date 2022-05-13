from exercise1 import run_exercise_1
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Demostrar que 5000 iteraciones son suficientes
    print("TESTING ITERATIONS")
    errors = []
    seeds = np.random.rand(25)*10000 #np.linspace(0, 10000, 25)
    iteration_options = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000]
    boxplot_index = 0
    for iterations in iteration_options:
        errors.append([])
        for seed in seeds:
            resp = run_exercise_1(50, 0, int(seed), iterations, 4, 1, False)
            errors[boxplot_index].append(resp[3])
        boxplot_index += 1
    plt.boxplot(errors, labels=iteration_options)
    plt.xlabel('Iteraciones')
    plt.ylabel('Error de Clasificación')
    plt.title("TP3-1 con Perceptrón")
    plt.show()
    plt.clf()

    # Demostrar que m=4 es suficiente (A veces da más alto a mayores, pero por dif mínima, mucho costo computacional)
    print("TESTING OPTIMAL SUPPORT POINTS")
    errors = []
    margins = []
    boxplot_index = 0
    m_values = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    for m in m_values:
        errors.append([])
        margins.append([])
        for seed in seeds:
           resp = run_exercise_1(50, 0, int(seed), 5000, m, 1, False)
           if len(resp)==8:
               errors[boxplot_index].append(resp[-1])
               margins[boxplot_index].append(resp[-2])
        boxplot_index += 1
    #plt.boxplot(errors, labels=m_values)
    #plt.xlabel('Puntos de Soporte')
    #plt.ylabel('Error de Clasificación')
    #plt.title("TP3-1 con Hiperplano Óptimo")
    #plt.show()
    #plt.clf()
    plt.boxplot(margins, labels=m_values)
    plt.xlabel('Puntos de Soporte')
    plt.ylabel('Margen')
    plt.title("TP3-1 con Hiperplano Óptimo")
    plt.show()
    plt.clf()

    print("TESTING SVM")
    errors = []
    margins = []
    svm_errors = []
    svm_margins = []
    boxplot_index = 0
    c_values = [0.01, 0.1, 1, 10, 100, 1000]
    for c in c_values:
        svm_errors.append([])
        svm_margins.append([])
        for seed in seeds:
            resp = run_exercise_1(50, 0, int(seed), 5000, 4, c, False)
            if len(resp)==8:
                svm_margins[boxplot_index].append(resp[4])
                svm_errors[boxplot_index].append(resp[5])
        boxplot_index += 1
    plt.boxplot(svm_errors, labels=c_values)
    plt.xlabel('C')
    plt.ylabel('Error de Clasificación')
    plt.title("TP3-1 con SVM")
    plt.show()
    plt.clf()
    plt.boxplot(svm_margins, labels=c_values)
    plt.xlabel('C')
    plt.ylabel('Margen')
    plt.title("TP3-1 con SVM")
    plt.show()
    plt.clf()


if __name__ == '__main__':
    main()
