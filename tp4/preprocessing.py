import argparse
from config.constants import Headers
from config.configurations import Configuration
from fileHandling import read_csv, export_csv
from preprocessing.analysis import analyze_dataset
from preprocessing.knn import replace_nearest_neighbor


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
    df = read_csv(args.file)

    # Show file analysis if very verbose
    if args.analyze:
        print("Running analysis...")
        analyze_dataset(df)
    
    # Perform file replacements
    print("Running replacements...")
    df = replace_nearest_neighbor(
        df, Headers.CHOLESTEROL.value, id_header=Headers.EXTRA_ID_HEADER.value, scaling_headers=[Headers.AGE.value, Headers.CAD_DUR.value],
        full_headers=[Headers.AGE.value, Headers.SEX.value, Headers.SIGDZ.value, Headers.TVDLM.value,
                      Headers.CAD_DUR.value, Headers.CHOLESTEROL.value, Headers.EXTRA_ID_HEADER.value],
        calculation_headers=[Headers.AGE.value, Headers.SEX.value, Headers.SIGDZ.value, Headers.TVDLM.value, Headers.CAD_DUR.value])
    
    # Store replacements in files
    if args.analyze:
        print("Running analysis...")
        analyze_dataset(df)
    export_csv(df, args.file_output, delimiter=';')


if __name__ == '__main__':
    main()
