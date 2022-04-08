from fileHandling import read_csv, print_entire_df

def run_exercise_1(filepath, cross_validation_k):
    df = read_csv(filepath)
    print_entire_df(df)