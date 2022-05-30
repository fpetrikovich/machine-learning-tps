from exercise1 import run_exercise_1
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Demostrar que 5000 iteraciones son suficientes
    print("TESTING MISCLASSIFICATION")
    errors = []
    margins = []
    seeds = np.random.rand(25)*10000 #np.linspace(0, 10000, 25)
    misclassifications = [1, 5, 10, 20, 30, 40, 50]
    boxplot_index = 0
    for misc in misclassifications:
        print("Miss = ", misc)
        errors.append([])
        margins.append([])
        for seed in seeds:
            resp = run_exercise_1(50, misc, int(seed), 5000, 4, 1, False)
            errors[boxplot_index].append(resp[3])
            margins[boxplot_index].append(resp[-2])
        boxplot_index += 1
    plt.boxplot(errors, labels=misclassifications)
    plt.xlabel('Puntos mal clasificados')
    plt.ylabel('Error de Clasificación')
    plt.title("TP3-1c con Perceptrón")
    plt.show()
    plt.clf()
    plt.boxplot(margins, labels=misclassifications)
    plt.xlabel('Puntos Mal Clasificados')
    plt.ylabel('Margen')
    plt.title("TP3-1c con Perceptrón")
    plt.show()
    plt.clf()

    print("TESTING SVM")
    c_values = [0.01, 1, 100]
    for c in c_values:
        svm_errors = []
        svm_margins = []
        boxplot_index = 0
        for misc in misclassifications:
            print("Misc = ", misc)
            svm_errors.append([])
            svm_margins.append([])
            for seed in seeds:
                resp = run_exercise_1(50, misc, int(seed), 5000, 4, c, False)
                svm_margins[boxplot_index].append(resp[4])
                svm_errors[boxplot_index].append(resp[5])
            boxplot_index += 1
        plt.boxplot(svm_errors, labels=misclassifications)
        plt.xlabel('Puntos Mal Clasificados')
        plt.ylabel('Error de Clasificación')
        title = "TP3-1d con SVM, C = " + str(c)
        plt.title(title)
        plt.show()
        plt.clf()

        plt.boxplot(svm_margins, labels=misclassifications)
        plt.xlabel('Puntos Mal Clasificados')
        plt.ylabel('Margen')
        plt.title(title)
        plt.show()
        plt.clf()


if __name__ == '__main__':
    main()
