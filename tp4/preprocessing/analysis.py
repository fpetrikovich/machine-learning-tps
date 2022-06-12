from fileHandling import print_entire_df
from config.configurations import Configuration
from config.constants import Headers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# https://stackoverflow.com/questions/9767241/setting-a-relative-frequency-in-a-matplotlib-histogram
def plot_histogram(df, header, bin_count, x_label):
    plt.hist(df[header], weights=np.zeros_like(df[header]) + 1. / df[header].size, bins=bin_count, color='red', ec="k")
    plt.xlabel(x_label)
    plt.ylabel('Frequency')
    plt.show()

def analyze_dataset(df):
    # Print entire DF if it's set to be very verbose
    if Configuration.isVeryVerbose():
        print_entire_df(df)

    # NaN values
    # Get DF with all NaN values
    nan_df = df[pd.isna(df[Headers.CHOLESTEROL.value])]
    print('---------------------------')
    print(f"NaN cholesterol values are {nan_df.shape[0]} out of {df.shape[0]} ({100 * nan_df.shape[0]/df.shape[0]}%)")
    print('---------------------------')
    
    # Get DF with all NaN values
    nan_df = df[pd.isna(df[Headers.TVDLM.value])]
    print('---------------------------')
    print(f"NaN cholesterol values are {nan_df.shape[0]} out of {df.shape[0]} ({100 * nan_df.shape[0]/df.shape[0]}%)")
    print('---------------------------')

    # Max values
    nan_free_df = df[~pd.isna(df[Headers.CHOLESTEROL.value])]
    print('---------------------------')
    print(f"Cholesterol values range from {np.min(nan_free_df[Headers.CHOLESTEROL.value])} to {np.max(nan_free_df[Headers.CHOLESTEROL.value])}")
    print(f"Cad.dur values range from {np.min(nan_free_df[Headers.CAD_DUR.value])} to {np.max(nan_free_df[Headers.CAD_DUR.value])}")
    print(f"Age values range from {np.min(nan_free_df[Headers.AGE.value])} to {np.max(nan_free_df[Headers.AGE.value])}")
    print('---------------------------')

    # Plotting cholesterol histogram
    plot_histogram(nan_free_df, Headers.CHOLESTEROL.value, 30, 'Cholesterol')
    
    # Plotting age histogram
    plot_histogram(nan_free_df, Headers.AGE.value, 30, 'Age')
    
    # Plotting cholesterol histogram
    plot_histogram(nan_free_df, Headers.CAD_DUR.value, 30, 'Cad.dur')
    
    # Plotting cholesterol histogram
    plot_histogram(nan_free_df, Headers.SEX.value, 2, 'Sex')
    
    # Plotting cholesterol histogram
    plot_histogram(nan_free_df, Headers.SIGDZ.value, 2, 'SIGDZ')
    
    # Plotting cholesterol histogram
    plot_histogram(nan_free_df, Headers.TVDLM.value, 2, 'TVDLM')