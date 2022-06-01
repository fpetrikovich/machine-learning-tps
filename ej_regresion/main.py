import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import Headers
import matplotlib.gridspec as gridspec

def print_entire_df (df):
    pd.set_option('display.max_rows', df.shape[0]+1)
    print(df)

def read_data(file):
    df = pd.read_csv(file)
    return df


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning Regression Exercise")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)
    args = parser.parse_args()

    df = read_data(args.file)
    print_entire_df(df)


if __name__ == '__main__':
    main()
