import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

def plot_confusion_matrix(matrix, headings):
    df_cm = pd.DataFrame(matrix, index = headings, columns = headings)
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    plt.show()