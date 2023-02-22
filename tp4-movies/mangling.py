import argparse
import math
import random

from numpy import NaN
from config.constants import Headers
from config.configurations import Configuration
from fileHandling import read_csv, export_csv, print_entire_df
from preprocessing.analysis import analyze_dataset
import json
import ast
import pandas as pd

def get_first_genre(genres):
    if len(genres) > 0:
        genres = genres.replace("\'", "\"")
        genres = json.loads(genres)
        return genres[0]['name'] if len(genres) > 0 else 0
    return 0

def count_collection(col):
    if len(col) > 0:
        # col = col.replace("\',", "\",").replace(" \'", " \"").replace("\':", "\":").replace("{\'", "{\"")
        col = ast.literal_eval(col)
        return len(col)
    return 0

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP4 - PREPROCESSING")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)    # Path to dataset
    parser.add_argument('-o', dest='file_output', required=True)    # Path to output
    parser.add_argument('-r', dest='repeat_ratio', required=True)    # Ratio of repeated rows, percent, 0 to 1
    parser.add_argument('-m', dest='missing_ratio', required=True)    # Ratio of rows with at least one missing item
    args = parser.parse_args()

    # Processing file
    df = read_csv(args.file, ";")
    cols = df.columns
    
    print("Missing data...")
    missing_prob = float(args.missing_ratio)
    for i in range(df.shape[0]):
        # If the probability hits, miss data
        if random.random() < missing_prob:
            df.at[i, cols[random.randint(0, len(cols) - 1)]] = NaN

    # Adding duplicates
    print("Adding duplicates...")
    total_new_rows = math.ceil(df.shape[0] * float(args.repeat_ratio))
    new_rows = []
    # Get all the new rows that will appear to create the DF only once
    for i in range(total_new_rows):
        new_rows.append(df.iloc[i])
    # Create data for DF
    new_data = {}
    for col in cols:
        new_data[col] = []
        for new_row in new_rows:
            new_data[col].append(new_row[col])
    # Merge DFs
    df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True, axis = 0)

    export_csv(df, args.file_output, delimiter=';')


if __name__ == '__main__':
    main()
