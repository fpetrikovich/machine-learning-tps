import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import Headers
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split

def print_entire_df (df):
    pd.set_option('display.max_rows', df.shape[0]+1)
    print(df)

def read_data(file):
    df = pd.read_csv(file, sep=' ')
    return df

def split_train_test_dataset(X, y):
    # 70% train, 30% test, 100 -> seed for shuffling
    return train_test_split(X, y, test_size = 0.3, random_state = 100)

def run_regression(df):
    X = df[[ Headers.BICROMIAL.value, Headers.PELVIC_BREADTH.value, Headers.BITROCHANTERIC.value, Headers.CHEST_DEPTH.value, Headers.CHEST_DIAM.value, Headers.ELBOW_DIAM.value, Headers.WRIST_DIAM.value, Headers.KNEE_DIAM.value, Headers.ANKLE_DIAM.value, Headers.SHOULDER_GIRTH.value, Headers.CHEST_GIRTH.value, Headers.WAIST_GIRTH.value, Headers.NAVEL_GIRTH.value, Headers.HIP_GIRTH.value, Headers.THIGH_GIRTH.value, Headers.BICEP_GIRTH.value, Headers.FOREARM_GIRTH.value, Headers.KNEE_GIRTH.value, Headers.CALF_GIRTH.value, Headers.ANKLE_GIRTH.value, Headers.WRIST_GIRTH.value, Headers.AGE.value, Headers.HEIGHT.value ]]
    y = df[Headers.WEIGHT.value]
    
    # Splitting dataset using sklearn
    x_train, x_test, y_train, y_test = split_train_test_dataset(X, y)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning Regression Exercise")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)
    args = parser.parse_args()

    df = read_data(args.file)

    run_regression(df)


if __name__ == '__main__':
    main()
