import argparse
from config.configurations import Configuration
from exerciseE import run_KMeans
from fileHandling import read_csv
from logistic import run_logistic

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP4")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)    # Path to dataset
    parser.add_argument('-p', dest='point', required=True)  # Exercise to run
    parser.add_argument('-mode', dest='mode', required=False)   # Running mode
    parser.add_argument('-crossk', dest='cross_k',
                        required=False)  # Cross validation
    parser.add_argument('-k', dest='group_k', required=False)  # K from Kmeans

    # The following are for Python 3.8 and under
    # Verbose, print or not
    parser.add_argument('-v', dest='verbose', action='store_true')
    parser.add_argument('-vv', dest='veryVerbose',
                        action='store_true')  # Verbose, print or not
    args = parser.parse_args()

    # Store configuration
    Configuration.setVerbose(args.verbose)
    Configuration.setVeryVerbose(args.veryVerbose)

    # Processing parameters
    item = args.point

    cross_k = None
    group_k = None
    
    file = args.file
    if args.cross_k != None:
        cross_k = int(args.cross_k)
    if args.group_k != None:
        group_k = int(args.group_k)

    # Processing file
    df = read_csv(args.file)

    print("[INFO] Running exercise", item, "...")
    if item == 'b':
        run_logistic(df, cross_k=cross_k, account_male_female=False)
    elif item == 'cd':
        run_logistic(df, cross_k=cross_k, account_male_female=True)
    elif item == 'e':
        if group_k is None: group_k = 2
        run_KMeans(file, group_k)


if __name__ == '__main__':
    main()
