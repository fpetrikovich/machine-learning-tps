import argparse

from exercise1 import run_exercise_1
from exercise2 import run_exercise_2
from exercise3 import run_exercise_3

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP0")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)
    parser.add_argument('-m', dest='mode', required=True)
    parser.add_argument('-p', dest='point', required=True)  # Ejercicio a ejecutar
    args = parser.parse_args()

    try:
        item = int(args.point)
        if item <= 0 or item > 3:
            print("[ERROR] Invalid value for input, must be between 1 and 3")
            exit(0)

        print("[INFO] Running exercise", item, "...")
        if item == 1:
            run_exercise_1(args.file)
        elif item == 2:
            run_exercise_2(args.file)
        elif item == 3:
            run_exercise_3(args.file)
    except:
        print("[ERROR] Invalid option input")

if __name__ == '__main__':
    main()
