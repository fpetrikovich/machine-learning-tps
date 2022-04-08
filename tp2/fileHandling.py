import pandas as pd

def read_excel(file):
    df = pd.read_excel(file)
    return df

def read_csv(file, delimiter = ';'):
    df = pd.read_csv(file, delimiter=delimiter)
    return df

def print_entire_df (df):
    pd.set_option('display.max_rows', df.shape[0]+1)
    pd.set_option('display.max_columns', None)
    print(df)