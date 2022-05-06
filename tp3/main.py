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

    # The following are for Python 3.8 and under
    parser.add_argument('-v', dest='verbose', action='store_true')  # Verbose, print or not
    parser.add_argument('-vv', dest='veryVerbose', action='store_true')  # Verbose, print or not
    args = parser.parse_args()
    item = int(args.point)

    # Store configuration
    Configuration.setVerbose(args.verbose)
    Configuration.setVeryVerbose(args.veryVerbose)

    print("[INFO] Running exercise", item, "...")
    if item == 1:
        run_exercise_1()
    elif item == 2:
        print("HOLA")
        #run_exercise_2(args.file)

if __name__ == '__main__':
    main()
