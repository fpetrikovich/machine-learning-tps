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

# Writes matrix to file 
# Input: file where to write, matrix to write
def write_matrix_to_file(filename, _matrix):
    mat = np.matrix(_matrix)
    with open('output/' + filename,'wb') as f:
        for line in mat:
            np.savetxt(f, line, fmt='%.2f')

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
    for h in headers:
        df[h] = (df[h] - np.min(df[h])) / (np.max(df[h]) - np.min(df[h]))
    # Add an extra ID header to the values
    df[extra_id_header] = np.arange(0, df.shape[0])
    return df

def split_dataset_data_labels(df, data_headers, label_headers):
    data = df[data_headers].reset_index(drop=True)
    labels = df[label_headers].reset_index(drop=True)
    return data, labels

def df_to_numpy(_df):
    return _df.copy(deep=True).to_numpy()
