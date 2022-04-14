from fileHandling import read_csv, print_entire_df
from constants import Ex1_Headers, EX1_DIVISION, Ex2_Modes, Ex2_Run, Tennis_Headers
import math

def show_analysis(df):
    print("Coming soon")

def run_cross_validation(df, cross_k):
    print("Coming soon")

def make_tree(training_set, goal_attribute):
    return ID3(training_set, goal_attribute, 0)

def ID3(df, goal_attribute, height):
    # STEPS 1-3: Create root
    possible_answers = df[goal_attribute].unique()
    if(len(possible_answers) == 1):
        return possible_answers[0]
    if len(df.columns) == 1:
        return df.mode()[goal_attribute][0]

    # STEP 4: Pick attribute
    gains = {}
    for attr in df.columns:
        if attr != goal_attribute:
            gains[attr] = gain(df, attr, goal_attribute)
    A = max(gains, key=gains.get)
    # TODO: If A's gain < u, we should create leaf node here
    tree = {'attribute': A, 'children': {}}

    #STEP 4.4:
    for vi in df[A].unique():
        # Get entries where it takes value vi, and remove A column
        subtree_df = df[df[A] == vi].loc[:, ~df.columns.isin([A])]
        tree['children'][vi] = ID3(subtree_df, goal_attribute, height+1)
    return tree

def H(df, goal_attribute):
    entropy = 0
    relative_frequencies = df[goal_attribute].value_counts(normalize=True)
    for p in relative_frequencies:
        entropy -= p * math.log(p, 2)
    return entropy

def HSv(df, goal_attribute, filter_attribute, filter_value):
    reduced_df = df[df[filter_attribute] == filter_value]
    return H(reduced_df, goal_attribute)

def gain(df, filter_attribute, goal_attribute):
    sum = 0
    relative_frequencies = df[filter_attribute].value_counts(normalize=True)
    for v in df[filter_attribute].unique():
        sum += relative_frequencies[v] * HSv(df, goal_attribute, filter_attribute, v)
    return H(df, goal_attribute) - sum

def draw_tree(tree, height):
    if type(tree) is dict:
        for i in range(height):
            print("| ", end="")
        print(str(tree['attribute']).upper())
        for child in tree['children'].keys():
            for i in range(height+1):
                print("| ", end="")
            print(child)
            draw_tree(tree['children'][child], height+2)
    else:
        for i in range(height-1):
            print("| ", end="")
        print("  ***",str(tree).upper(),"***")

def dicretize_data(df):
    # CREDIT
    quantiles = [.1, .25, .50, .75, .9]
    credit_thresholds = [0]
    for value in df[Ex1_Headers.CREDIT_AMOUNT.value].quantile(quantiles):
        credit_thresholds.append(int(value))
    new_header = 'aux'
    for i in range(len(credit_thresholds)-1):
        df.loc[df[Ex1_Headers.CREDIT_AMOUNT.value].between(credit_thresholds[i], credit_thresholds[i+1], inclusive="left"),
            new_header] = '$'+str(credit_thresholds[i])+"-"+str(credit_thresholds[i+1])
    df.loc[df[Ex1_Headers.CREDIT_AMOUNT.value] >= credit_thresholds[len(credit_thresholds)-1], new_header] = 'Mayor a $'+str(credit_thresholds[len(credit_thresholds)-1])
    df = df.drop(columns=[Ex1_Headers.CREDIT_AMOUNT.value]).rename(columns={new_header: Ex1_Headers.CREDIT_AMOUNT.value})

    # AGES
    age_ceilings = [0, 25, 40, 65]
    for i in range(len(age_ceilings)-1):
        df.loc[df[Ex1_Headers.AGE.value].between(age_ceilings[i], age_ceilings[i+1], inclusive="left"),
            new_header] = str(age_ceilings[i])+"-"+str(age_ceilings[i+1])+" años"
    df.loc[df[Ex1_Headers.AGE.value] >= age_ceilings[len(age_ceilings)-1], new_header] = str(age_ceilings[len(age_ceilings)-1])+"+ años"
    df = df.drop(columns=[Ex1_Headers.AGE.value]).rename(columns={new_header: Ex1_Headers.AGE.value})

    # LENGTH
    quantiles = [.1, .25, .50, .75, .9]
    duration_ceilings = [0]
    for value in df[Ex1_Headers.CREDIT_DURATION.value].quantile(quantiles):
        duration_ceilings.append(int(value))
    for i in range(len(duration_ceilings)-1):
        df.loc[df[Ex1_Headers.CREDIT_DURATION.value].between(duration_ceilings[i], duration_ceilings[i+1], inclusive="left"),
            new_header] = str(duration_ceilings[i])+"-"+str(duration_ceilings[i+1])+" meses"
    df.loc[df[Ex1_Headers.CREDIT_DURATION.value] >= duration_ceilings[len(duration_ceilings)-1], new_header] = str(duration_ceilings[len(duration_ceilings)-1])+'+ meses'
    df = df.drop(columns=[Ex1_Headers.CREDIT_DURATION.value]).rename(columns={new_header: Ex1_Headers.CREDIT_DURATION.value})
    return df

def run_exercise_1(filepath, cross_validation_k=None, solve_mode=Ex2_Run.SOLVE):
    df = read_csv(filepath, ',')
    df = dicretize_data(df)
    #print_entire_df(df)

    if solve_mode == Ex2_Run.ANALYZE:
        if Configuration.isVerbose():
            print_entire_df(df)
        show_analysis(df)
    else:
        # Shuffle df
        df = df.sample(frac=1)
        if cross_validation_k == None:
            # Divide dataset
            train_number = int(len(df)/EX1_DIVISION) * (EX1_DIVISION - 1)
            train = df.iloc[0:train_number]
            test = df.iloc[train_number+1:len(df)]
            tree = make_tree(train, Ex1_Headers.CREDITABILITY.value)
            draw_tree(tree, 0)
            #perform_classification(train, test, mode=mode)
        else:
            run_cross_validation(df=df, cross_k=cross_validation_k)
