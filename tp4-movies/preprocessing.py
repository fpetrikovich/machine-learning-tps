import argparse
from config.constants import Headers
from config.configurations import Configuration
from fileHandling import read_csv, export_csv, print_entire_df
from preprocessing.analysis import analyze_dataset
import json
import ast

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

    # The following are for Python 3.8 and under
    # Verbose, print or not
    parser.add_argument('-analyze', dest='analyze', action='store_true')
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-vv', dest='veryVerbose',
                        action='store_true')  # Verbose, print or not
    args = parser.parse_args()

    # Store configuration
    Configuration.setVerbose(args.verbose)
    Configuration.setVeryVerbose(args.veryVerbose)

    # Processing file
    df = read_csv(args.file, ",")
    
    # Perform file replacements
    print("Running replacements...")
    df[Headers.POPULARITY.value] = df[Headers.POPULARITY.value].astype(float)

    # Filtering
    filtered_df = df
    # Keep movies that have a budget
    filtered_df = filtered_df[filtered_df[Headers.BUDGET.value] > 0]
    # Keep movies that have revenue
    filtered_df = filtered_df[filtered_df[Headers.REVENUE.value] > 10000]
    filtered_df = filtered_df[filtered_df[Headers.RUNTIME.value] > 0]
    filtered_df = filtered_df[filtered_df[Headers.POPULARITY.value] < 60]
    filtered_df = filtered_df[filtered_df[Headers.STATUS.value] == 'Released']
    print("Shape after filters", filtered_df.shape)

    # Keeping only some columns
    filtered_df = filtered_df.drop(Headers.ADULT.value, axis=1)
    filtered_df = filtered_df.drop(Headers.COLLECTION.value, axis=1)
    filtered_df = filtered_df.drop(Headers.HOMEPAGE.value, axis=1)
    filtered_df = filtered_df.drop(Headers.TAGLINE.value, axis=1)
    filtered_df = filtered_df.drop(Headers.TITLE.value, axis=1)
    filtered_df = filtered_df.drop(Headers.STATUS.value, axis=1)
    filtered_df = filtered_df.drop(Headers.VIDEO.value, axis=1)
    filtered_df = filtered_df.drop(Headers.POSTER.value, axis=1)
    filtered_df = filtered_df.drop(Headers.LANGUAGE.value, axis=1)
    filtered_df = filtered_df.drop(Headers.ID.value, axis=1)

    # Modifying some columns
    filtered_df[Headers.GENRES.value] = filtered_df.apply(lambda row : get_first_genre(row[Headers.GENRES.value]), axis = 1)
    filtered_df[Headers.PRODUCTION.value] = filtered_df.apply(lambda row : count_collection(row[Headers.PRODUCTION.value]), axis = 1)
    filtered_df[Headers.PRODUCTION_COUNTRY.value] = filtered_df.apply(lambda row : count_collection(row[Headers.PRODUCTION_COUNTRY.value]), axis = 1)
    filtered_df[Headers.LANGUAGE_SPOKEN.value] = filtered_df.apply(lambda row : count_collection(row[Headers.LANGUAGE_SPOKEN.value]), axis = 1)
    filtered_df = filtered_df[filtered_df[Headers.GENRES.value] != 0]
    print("Shape after filters", filtered_df.shape)
    print_entire_df(filtered_df[filtered_df[Headers.RUNTIME.value] >= 280])

    # Store replacements in files
    if args.analyze:
        print("Running analysis...")
        analyze_dataset(filtered_df)
    export_csv(filtered_df, args.file_output, delimiter=';')


if __name__ == '__main__':
    main()
