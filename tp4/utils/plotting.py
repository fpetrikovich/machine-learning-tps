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
    fig = plt.figure(figsize = (10,7))
    plt.tick_params(axis='both', which='major', labelsize=10, labelbottom = False, bottom=False, top = False, labeltop=True)
    ax = sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}, fmt='g') # font size
    ax.xaxis.tick_top() # x axis on top
    plt.savefig(filename)
    plt.close('all')
    plt.close(fig)

def plot_hierarchy(dendogram, X_test, y_test):
    class1 = [l.tolist() for l in dendogram['left']['content']]
    class2 = [l.tolist() for l in dendogram['right']['content']]
    sizes = [60]*(len(class1)+len(class2))

    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
    sctt = ax.scatter3D(X_test[:,0], X_test[:,1], X_test[:,2], sizes, alpha=0.8, c=y_test, marker ='x')
    for p in class1:
        ax.scatter3D(p[0], p[1], p[2], c='blue', alpha=0.8)
    for p in class2:
        ax.scatter3D(p[0], p[1], p[2], c='red', alpha=0.8)
    plt.show()
