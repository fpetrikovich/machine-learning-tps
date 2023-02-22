import argparse
from config.configurations import Configuration
from exerciseE import run_KMeans, run_hierarchy, run_kohonen
from pca import apply_pca
from plotting import plot_3d
from fileHandling import read_csv

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP4 - Movies Edition")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)    # Path to dataset
    parser.add_argument('-p', dest='point', required=True)  # Exercise to run

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

    n_kohonen = None 
    iterations = None 
    file = args.file
    if args.n_kohonen != None:
        n_kohonen = int(args.n_kohonen)
    if args.iterations != None:
        iterations = int(args.iterations)

    # Processing file
    df = read_csv(args.file)

    print("[INFO] Running exercise", item, "...")
    if item == 'means':
        run_KMeans(file)
    elif item == 'hier':
        run_hierarchy(file)
    elif item == 'koho':
        run_kohonen(file, n_kohonen, iterations)
    elif item == 'pca':
        apply_pca(file)
    elif item == 'plot':
        plot_3d(file)


if __name__ == '__main__':
    main()
