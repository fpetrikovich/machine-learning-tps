import argparse

from exercise1 import run_exercise_1
from exercise1RF import run_exercise_1_rf
from exercise2 import run_exercise_2
from configurations import Configuration
from constants import Ex2_Modes, Ex2_Run
from parameters import Parameters

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP2")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)   # Archivo para usar
    parser.add_argument('-p', dest='point', required=True)  # Ejercicio a ejecutar
    parser.add_argument('-m', dest='mode')  # Modo
    parser.add_argument('-sm', dest='solve_mode')  # Modo
    parser.add_argument('-crossk', dest='cross_k')  # Modo
    parser.add_argument('-k', dest='k_neighbors')  # Modo
    parser.add_argument('-ss', dest='sample_size')  # Sample size for RF
    parser.add_argument('-ns', dest='number_sample')  # Number of samples for RF
    parser.add_argument('-hl', dest='height_limit')  # Max height of tree
    parser.add_argument('-gu', dest='gain_umbral')  
    parser.add_argument('-eu', dest='example_umbral') 
    parser.add_argument('-sal', dest='split_attr_limit')  

    # The following two are if using Python 3.9 and up
    #parser.add_argument('-v', dest='verbose', action=argparse.BooleanOptionalAction, default=False)  # Verbose, print or not
    #parser.add_argument('-vv', dest='veryVerbose', action=argparse.BooleanOptionalAction, default=False)  # Verbose, print or not

    # The following are for Python 3.8 and under
    parser.add_argument('-v', dest='verbose', action='store_true')  # Verbose, print or not
    parser.add_argument('-vv', dest='veryVerbose', action='store_true')  # Verbose, print or not
    args = parser.parse_args()

    cross_k, mode, solve_mode = None, None, None
    # try:
    item = int(args.point)
    if args.cross_k != None:
        cross_k = int(args.cross_k)
    if item <= 0 or item > 3:
        print("[ERROR] Invalid value for input, must be between 1 and 3")
        exit(0)
    if item == 1:
        if args.mode == Ex2_Run.ANALYZE.value:
            mode = Ex2_Run.ANALYZE
        else:
            mode = Ex2_Run.SOLVE
    if item == 2:
        if args.mode == Ex2_Modes.WEIGHTED.value:
            mode = Ex2_Modes.WEIGHTED
        else:
            mode = Ex2_Modes.SIMPLE
        if args.solve_mode == Ex2_Run.ANALYZE.value:
            solve_mode = Ex2_Run.ANALYZE
        else:
            solve_mode = Ex2_Run.SOLVE
        k_neighbors = int(args.k_neighbors)
    if item == 3:
        Parameters.set_random_forest_values(
            args.example_umbral,
            args.gain_umbral,
            args.height_limit,
            args.split_attr_limit,
            args.sample_size,
            args.number_sample
        )
        if args.mode == Ex2_Run.ANALYZE.value:
            mode = Ex2_Run.ANALYZE
        else:                                                         
            mode = Ex2_Run.SOLVE
    # except:
    #     print("[ERROR] Invalid option input")
    #     exit(0)

    # Store configuration
    Configuration.setVerbose(args.verbose)
    Configuration.setVeryVerbose(args.veryVerbose)

    print("[INFO] Running exercise", item, "...")
    if item == 1:
        run_exercise_1(args.file, cross_validation_k=cross_k, mode=mode)
    elif item == 2:
        run_exercise_2(args.file, mode=mode, k_neighbors=k_neighbors, cross_validation_k=cross_k, solve_mode=solve_mode)
    else:
        run_exercise_1_rf(args.file, cross_validation_k=cross_k, mode=mode)

if __name__ == '__main__':
    main()
