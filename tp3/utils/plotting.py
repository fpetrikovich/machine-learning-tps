import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix, headings):
    df_cm = pd.DataFrame(matrix, index = headings, columns = headings)
    plt.figure(figsize = (10,7))
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    ax.xaxis.tick_top() # x axis on top
    plt.show()

def save_confusion_matrix(matrix, headings, filename):
    df_cm = pd.DataFrame(matrix, index = headings, columns = headings)
    plt.figure(figsize = (10,7))
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    ax.xaxis.tick_top() # x axis on top
    plt.savefig(filename)