import argparse
from xmlrpc.client import boolean

from exercise1 import run_exercise_1
from exercise2 import run_exercise_2
from exercise3 import run_exercise_3
from configurations import Configuration

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP0")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)   # Archivo para usar
    parser.add_argument('-p', dest='point', required=True)  # Ejercicio a ejecutar
    parser.add_argument('-m', dest='mode')  # Modo
    parser.add_argument('-wc', dest='word_count')  # Modo
    parser.add_argument('-k', dest='cross_k')  # Modo

    # The following two are if using Python 3.9 and up
    #parser.add_argument('-v', dest='verbose', action=argparse.BooleanOptionalAction, default=False)  # Verbose, print or not
    #parser.add_argument('-vv', dest='veryVerbose', action=argparse.BooleanOptionalAction, default=False)  # Verbose, print or not

    # The following are for Python 3.8 and under
    parser.add_argument('-v', dest='verbose', action='store_true')  # Verbose, print or not
    parser.add_argument('-vv', dest='veryVerbose', action='store_true')  # Verbose, print or not
    parser.add_argument('-roc', dest='roc', action='store_true')  # Verbose, print or not
    args = parser.parse_args()

    word_count = 250
    cross_k = None
    try:
        item = int(args.point)
        if args.word_count != None:
            word_count = int(args.word_count)
        if args.cross_k != None:
            cross_k = int(args.cross_k)
        if item <= 0 or item > 3:
            print("[ERROR] Invalid value for input, must be between 1 and 3")
            exit(0)
    except:
        print("[ERROR] Invalid option input")
        exit(0)

    # Store configuration
    Configuration.setVerbose(args.verbose)
    Configuration.setVeryVerbose(args.veryVerbose)

    print("[INFO] Running exercise", item, "...")
    if item == 1:
        run_exercise_1(args.file)
    elif item == 2:
        run_exercise_2(args.file, args.mode, word_count, cross_k, args.roc)
    elif item == 3:
        run_exercise_3(args.file)

if __name__ == '__main__':
    main()
