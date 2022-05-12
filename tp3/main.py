import argparse
from exercise1 import run_exercise_1
#from exercise2 import run_exercise_2
from config.configurations import Configuration
from config.parameters import Parameters

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP3")

    # Add arguments
    #parser.add_argument('-f', dest='file', required=True)   # Archivo para usar
    parser.add_argument('-p', dest='point', required=True)
    parser.add_argument('-i', dest='iterations', required=False)
    parser.add_argument('-n', dest='n', required=False)
    parser.add_argument('-m', dest='m', required=False)
    parser.add_argument('-s', dest='seed', required=False)
    parser.add_argument('-c', dest='C', required=False)

    # The following are for Python 3.8 and under
    parser.add_argument('-v', dest='verbose', action='store_true')  # Verbose, print or not
    parser.add_argument('-vv', dest='veryVerbose', action='store_true')  # Verbose, print or not
    args = parser.parse_args()
    n=25
    seed = None
    iterations = 100
    m = 4
    C = 1
    item = int(args.point)
    if args.n:
        n = int(args.n)
    if args.seed:
        seed = int(args.seed)
    if args.iterations:
        iterations = int(args.iterations)
    if args.m:
        m = int(args.m)
    if args.C:
        C = int(args.C)

    # Store configuration
    Configuration.setVerbose(args.verbose)
    Configuration.setVeryVerbose(args.veryVerbose)

    print("[INFO] Running exercise", item, "...")
    if item == 1:
        run_exercise_1(n, seed, iterations, m, C)
    elif item == 2:
        print("HOLA")
        #run_exercise_2(args.file)

if __name__ == '__main__':
    main()
