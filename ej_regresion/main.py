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

def split_df_all_attributes(df):
    x = df[[ Headers.BICROMIAL.value, Headers.PELVIC_BREADTH.value, Headers.BITROCHANTERIC.value, Headers.CHEST_DEPTH.value,
        Headers.CHEST_DIAM.value, Headers.ELBOW_DIAM.value, Headers.WRIST_DIAM.value, Headers.KNEE_DIAM.value, Headers.ANKLE_DIAM.value,
        Headers.SHOULDER_GIRTH.value, Headers.CHEST_GIRTH.value, Headers.WAIST_GIRTH.value, Headers.NAVEL_GIRTH.value, Headers.HIP_GIRTH.value,
        Headers.THIGH_GIRTH.value, Headers.BICEP_GIRTH.value, Headers.FOREARM_GIRTH.value, Headers.KNEE_GIRTH.value, Headers.CALF_GIRTH.value,
        Headers.ANKLE_GIRTH.value, Headers.WRIST_GIRTH.value, Headers.AGE.value, Headers.HEIGHT.value ]]
    y = df[Headers.WEIGHT.value]
    x_train, x_test, y_train, y_test = split_train_test_dataset(x, y)
    return x_train, x_test, y_train, y_test

def run_regression(x_train, y_train):
    # Convert X and Y to matrix and array
    matrix = []
    for n in range(len(x_train)):
        matrix.append([1])
        for p in range(len(x_train.columns)):
            matrix[-1].append(x_train.iloc[n][p])
    X = np.matrix(np.array(matrix))
    Y = []
    for i in range(len(y_train)):
        Y.append([y_train.iloc[i]])
    Y = np.array(Y)

    # Calculate betas
    Xt = X.T
    B = (Xt * X).I * Xt * Y
    return B, X

def predict(input, B):
    Yi = B[0,0]
    for j in range(len(input)):
        Yi += B[j+1, 0] * input[j]
    return Yi

def test(x_test, y_test, B):
    RSS = 0
    n = len(x_test)
    p = len(x_test.iloc[0])
    for i in range(n):
        actual = predict(x_test.iloc[i], B)
        expected = y_test.iloc[i]
        diff = abs(actual-expected)
        RSS += diff**2
    sigma2 = RSS/(n-p-1)
    return RSS, sigma2

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning Regression Exercise")
    # Add arguments
    parser.add_argument('-f', dest='file', required=True)
    args = parser.parse_args()
    df = read_data(args.file)
    x_train, x_test, y_train, y_test = split_df_all_attributes(df)
    B, X = run_regression(x_train, y_train)
    RSS, sigma2 = test(x_test, y_test, B)
    varB = sigma2 * (X.T*X).I
    print("RSS =", RSS)
    print("sigma2 =", sigma2)
    #print("Var(B) =", varB)

if __name__ == '__main__':
    main()
