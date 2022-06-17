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

def plot_hierarchy(classes, X_train, X_test, y_test):
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
    ax.scatter3D(X_test[:,0], X_test[:,1], X_test[:,2], alpha=0.8, c=y_test[:,0], marker ='x')
    for c in classes:
        points = X_train[c]
        ax.scatter3D(points[:,0], points[:,1], points[:,2], alpha=0.8)
    plt.show()

def plot_save_hierarchy(classes, X_train, X_test, y_test, filename):
    fig = plt.figure(figsize = (16, 9))
    ax = plt.axes(projection ="3d")
    ax.grid(b = True, color ='grey', linestyle ='-.', linewidth = 0.3, alpha = 0.2)
    ax.scatter3D(X_test[:,0], X_test[:,1], X_test[:,2], alpha=0.8, c=y_test[:,0], marker ='x')
    for c in classes:
        points = X_train[c]
        ax.scatter3D(points[:,0], points[:,1], points[:,2], alpha=0.8)
    plt.savefig(filename)
    plt.close('all')
    plt.close(fig)
