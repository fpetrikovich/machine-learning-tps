import argparse
from exercise1 import run_exercise_1
from exercise2 import run_exercise_2
from config.configurations import Configuration
from config.parameters import Parameters

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP3")

    # Add arguments
    parser.add_argument('-f', dest='folder', required=False)    # Path to folder with images
    parser.add_argument('-p', dest='point', required=True)
    parser.add_argument('-i', dest='iterations', required=False)
    parser.add_argument('-n', dest='n', required=False)
    parser.add_argument('-mc', dest='misclassifications', required=False)
    parser.add_argument('-m', dest='m', required=False)
    parser.add_argument('-s', dest='seed', required=False)
    parser.add_argument('-c', dest='C', required=False)
    parser.add_argument('-ker', dest='kernel', required=False)
    parser.add_argument('-kopt', dest='kernel_options', required=False)
    parser.add_argument('-k', dest='cross_k', required=False)
    parser.add_argument('-mode', dest='mode', required=False)

    # The following are for Python 3.8 and under
    parser.add_argument('-v', dest='verbose', action='store_true')  # Verbose, print or not
    parser.add_argument('-vv', dest='veryVerbose', action='store_true')  # Verbose, print or not
    args = parser.parse_args()
    n=25
    seed = None
    iterations = 5000
    misclassifications = 0
    kernel = None
    m = 4
    C = 10
    item = int(args.point)
    if args.misclassifications:
        misclassifications = int(args.misclassifications)
    if args.n:
        n = int(args.n)
    if args.seed:
        seed = int(args.seed)
    if args.iterations:
        iterations = int(args.iterations)
    if args.m:
        m = int(args.m)
    if args.C:
        C = float(args.C)
    if args.kernel:
        kernel = args.kernel
    # Cross validation k
    cross_k = None
    if args.cross_k:
        cross_k = int(args.cross_k)
    # Kernel options
    ker_opts = None
    if args.kernel_options:
        ker_opts = float(args.kernel_options)

    # Store configuration
    Configuration.setVerbose(args.verbose)
    Configuration.setVeryVerbose(args.veryVerbose)

    print("[INFO] Running exercise", item, "...")
    if item == 1:
        run_exercise_1(n, misclassifications, seed, iterations, m, C)
    elif item == 2:
        if args.folder == None:
            print('[ERROR] Missing path folder with images')
            return
        run_exercise_2(args.folder, svm_c=C, svm_kernel=kernel, cross_k=cross_k, mode=args.mode, kernel_options=ker_opts)

if __name__ == '__main__':
    main()
