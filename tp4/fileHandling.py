import pandas as pd
import numpy as np

####################################################################################
################################## READING DATA ####################################
####################################################################################

def read_excel(file):
    df = pd.read_excel(file)
    return df

def read_csv(file, delimiter = ';'):
    df = pd.read_csv(file, delimiter=delimiter)
    return df

####################################################################################
################################# WRITING DATA #####################################
####################################################################################

def export_csv(df, file, delimiter = ';'):
    df.to_csv (file, index = None, header=True, sep=delimiter) 

####################################################################################
################################# PRINTING DATA ####################################
####################################################################################

def print_entire_df(df):
    pd.set_option('display.max_rows', df.shape[0]+1)
    pd.set_option('display.max_columns', None)
    print(df)

####################################################################################
############################### DATASET OPERATIONS #################################
####################################################################################

def shuffle_dataset(df):
    return df.sample(frac=1)

def split_dataset(df, percentage):
    train_number = int(df.shape[0] * percentage)
    train = df.iloc[0:train_number]
    test = df.iloc[train_number:len(df)]
    return train, test

# Scales the df given the target header list
def scale_df(_df, headers, extra_id_header):
    # Create a copy just in case
    df = _df.copy(deep=True)
    # Iterate headers to modify
    for header in headers:
        h = header.value
        df[h] = (df[h] - np.min(df[h])) / (np.max(df[h]) - np.min(df[h]))
    # Add an extra ID header to the values
    df[extra_id_header] = np.arange(0, df.shape[0])
    return df