import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import Headers, Modes
import matplotlib.gridspec as gridspec

def print_entire_df (df):
    pd.set_option('display.max_rows', df.shape[0]+1)
    print(df)

def read_data(file):
    df = pd.read_excel(file)
    return df

def process_replace(df):
    return df.replace(to_replace=999.99, value=np.nan)

# Filter all the N/A rows
# Replace all 999.99 values in the column with the calculation for that column
def process_mean(df):
    for header in Headers:
        # Not process Sexo
        if header != Headers.SEXO:
            h = header.value
            filter = df[h] != 999.99
            df[h] = df[h].replace(to_replace=999.99, value=np.mean(df[filter][h]))
    return df

def process_median(df):
    for header in Headers:
        if header != Headers.SEXO:
            h = header.value
            filter = df[h] != 999.99
            df[h] = df[h].replace(to_replace=999.99, value=np.median(df[filter][h]))
    return df

# Processing changes according to given mode
def process_data(df, mode):
    if mode == Modes.REMOVE.name:
        df = process_replace(df)
    elif mode == Modes.MEDIAN.name:
        df = process_mean(df)
    elif mode == Modes.MEAN.name:
        df = process_median(df)
    return df


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP0")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)
    parser.add_argument('-m', dest='mode', required=True)
    args = parser.parse_args()

    df = read_data(args.file)
    new_df = process_data(df, args.mode)
    print_entire_df(new_df)

    # Create BoxPlot and Scatterplot graphs
    for header in Headers:
        if header != Headers.SEXO:
            h = header.value


if __name__ == '__main__':
    main()
