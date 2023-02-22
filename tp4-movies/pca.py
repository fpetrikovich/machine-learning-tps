from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from fileHandling import read_csv, df_to_numpy, print_entire_df
from config.constants import Headers, Similarity_Methods
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def apply_pca(file):
    df = read_csv(file)
    df = df[(df[Headers.GENRES.value] == 'Drama') | (df[Headers.GENRES.value] == 'Comedy') | (df[Headers.GENRES.value] == 'Action')]
    # features = [Headers.BUDGET.value, Headers.REVENUE.value]
    # features = [Headers.POPULARITY.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value, Headers.BUDGET.value, Headers.REVENUE.value]
    features = [Headers.POPULARITY.value, Headers.RUNTIME.value, Headers.VOTE_AVG.value, Headers.BUDGET.value, Headers.REVENUE.value, Headers.PRODUCTION.value, Headers.PRODUCTION_COUNTRY.value, Headers.LANGUAGE_SPOKEN.value]
    target_column = Headers.GENRES.value
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target_column
    y = df.loc[:,[target_column]].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2'])
    finalDf = pd.concat([principalDf, df[[target_column]]], axis = 1)
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(1,1,1) 
    ax.set_xlabel('PC1', fontsize = 15)
    ax.set_ylabel('PC2', fontsize = 15)
    ax.set_title('2 component PCA', fontsize = 20)
    targets = ['Drama', 'Comedy', 'Action']
    colors = ['r', 'g', 'b']
    for target, color in zip(targets,colors):
        indicesToKeep = finalDf[target_column] == target
        ax.scatter(finalDf.loc[indicesToKeep, 'PC1']
                , finalDf.loc[indicesToKeep, 'PC2']
                , c = color
                , s = 50)
    ax.legend(targets)
    ax.grid()
    plt.show()

    print(pd.DataFrame(pca.components_,columns=features,index = ['PC-1','PC-2']))
    X_axis = np.arange(len(features))
    plt.barh(X_axis - 0.2, pca.components_[0], label="PC1", height=0.4)
    plt.barh(X_axis + 0.2, pca.components_[1], label="PC2", height=0.4)
    plt.yticks(X_axis, features)
    plt.xlabel("Importance")
    plt.title("Feature Importance")
    plt.legend()
    plt.show()
