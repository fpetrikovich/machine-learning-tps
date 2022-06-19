import argparse
from config.configurations import Configuration
from exerciseE import run_KMeans, run_hierarchy, run_kohonen
from fileHandling import read_csv
from logistic import run_logistic

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP4")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)    # Path to dataset
    parser.add_argument('-ftest', dest='file_test', required=False)    # Path to dataset
    parser.add_argument('-p', dest='point', required=True)  # Exercise to run
    parser.add_argument('-mode', dest='mode', required=False)   # Running mode
    parser.add_argument('-crossk', dest='cross_k',
                        required=False)  # Cross validation
    parser.add_argument('-k', dest='group_k', required=False)  # K from Kmeans
    parser.add_argument('-n', dest='n_kohonen', required=False)  # N para tamano de red de kohonen
    parser.add_argument('-it', dest='iterations', required=False)  # iteraciones de kohonen

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
    df_test = None
    n_kohonen = None 
    iterations = None 
    file = args.file
    if args.cross_k != None:
        cross_k = int(args.cross_k)
    if args.group_k != None:
        group_k = int(args.group_k)
    if args.n_kohonen != None:
        n_kohonen = int(args.n_kohonen)
    if args.iterations != None:
        iterations = int(args.iterations)
    if args.file_test != None:
        df_test = read_csv(args.file_test)

    # Processing file
    df = read_csv(args.file)

    print("[INFO] Running exercise", item, "...")
    if item == 'b':
        run_logistic(df, cross_k=cross_k, account_male_female=False, df_test = df_test)
    elif item == 'cd':
        run_logistic(df, cross_k=cross_k, account_male_female=True, df_test = df_test)
    elif item == 'e':
        if group_k is None: group_k = 2
        run_KMeans(file, group_k)
    elif item == 'f':
        run_hierarchy(file)
    elif item == 'g':
        run_kohonen(file, n_kohonen, iterations)


if __name__ == '__main__':
    main()
