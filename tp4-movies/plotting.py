from cProfile import label
from sklearn.preprocessing import StandardScaler
from fileHandling import read_csv, df_to_numpy, print_entire_df
from config.constants import Headers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_3d(file):
    df = read_csv(file)
    df = df[(df[Headers.GENRES.value] == 'Drama') | (df[Headers.GENRES.value] == 'Comedy') | (df[Headers.GENRES.value] == 'Action')]
    # features = [Headers.BUDGET.value, Headers.REVENUE.value]
    # features = [Headers.POPULARITY.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value, Headers.BUDGET.value, Headers.REVENUE.value]
    features = [Headers.POPULARITY.value, Headers.BUDGET.value, Headers.REVENUE.value]
    target_column = Headers.GENRES.value
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    for cat in ['Drama', 'Comedy', 'Action']:
        filtered_df = df[(df[Headers.GENRES.value] == cat)]
        # Separating out the features
        x = filtered_df.loc[:, features].values
        # Standardizing the features
        x = StandardScaler().fit_transform(x)
        ax.scatter3D(x[:,0], x[:,1], x[:,2], label = cat)
    plt.show()
