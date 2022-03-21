import pandas as pd

def read_data(file):
    df = pd.read_excel(file)
    return df

def print_entire_df (df):
    pd.set_option('display.max_rows', df.shape[0]+1)
    print(df)