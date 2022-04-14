from fileHandling import read_csv, print_entire_df
from constants import Ex1_Headers, EX1_DIVISION, Ex2_Modes, Ex2_Run, Tennis_Headers
from confusion import get_accuracy, get_precision
from plotting import plot_confusion_matrix
from configurations import Configuration
import matplotlib.pyplot as plt
import numpy as np
import datetime
import math

EXAMPLES_UMBRAL = 5
GAIN_UMBRAL = 0.05
HEIGHT_LIMIT = 8
node_counter = 0

def show_analysis(df):
    print("Coming soon")

def run_cross_validation(df, cross_k):
    print("Coming soon")

def make_tree(df, training_set, goal_attribute):
    map = {}
    for attr in df.columns:
        map[attr] = []
        for vi in df[attr].unique():
            map[attr].append(vi)
    return ID3(training_set, goal_attribute, map, 0, None)

def ID3(df, goal_attribute, attrs_and_values, height, parent_mode):
    # STEPS 1-3: Create root
    if df.empty:
        return parent_mode
    mode = df[goal_attribute].mode()[0]
    possible_answers = df[goal_attribute].unique()
    if(len(possible_answers) == 1): # Only one answer
        return mode
    if len(df.columns) == 1 or df.shape[0] < EXAMPLES_UMBRAL: # No more info, or very few entries
        return mode

    # STEP 4: Pick attribute
    gains = {}
    for attr in df.columns:
        if attr != goal_attribute:
            gains[attr] = gain(df, attr, goal_attribute)
    A = max(gains, key=gains.get)

    # TRIMMING
    if gains[A] < GAIN_UMBRAL:
        return mode
    if height > HEIGHT_LIMIT:
        return mode

    #STEP 4.4:
    tree = {'attribute': A, 'children': {}}
    for vi in attrs_and_values[A]:
        # Get entries where it takes value vi, and remove A column
        subtree_df = df[df[A] == vi].loc[:, ~df.columns.isin([A])]
        tree['children'][vi] = ID3(subtree_df, goal_attribute, attrs_and_values, height+1, mode)
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

def export_tree(tree):
    global node_counter
    node_counter = 0
    dot_file = open("tree.dot", "w")
    dot_file.write("digraph {\nsize = \"10,20!\";\nratio = \"fill\";\nrankdir=\"LR\";overlap=false;\n")
    draw_tree_dot(tree, dot_file)
    dot_file.write("}")
    dot_file.close()

def draw_tree_text(tree, height):
    if type(tree) is dict:
        for i in range(height):
            print("| ", end="")
        print(str(tree['attribute']).upper())
        for child in tree['children'].keys():
            for i in range(height+1):
                print("| ", end="")
            print(child)
            draw_tree_rec(tree['children'][child], height+2)
        return node_name
    else:
        for i in range(height-1):
            print("| ", end="")
        print("  ***",str(tree).upper(),"***")

def draw_tree_dot(tree, dot_file):
    global node_counter
    node_name = "n"+str(node_counter)
    if type(tree) is dict:
        dot_file.write("\t"+node_name +" [ fontsize=30 shape=\"box\" label=\"" +str(tree['attribute']).upper() +"\" ]\n")
        node_counter += 1
        for child in sorted(tree['children'].keys()):
            child_node_name = draw_tree_dot(tree['children'][child], dot_file)
            dot_file.write("\t"+node_name +" -> " +child_node_name +" [ fontsize=20 xlabel=\"" +str(child) +"\" ]\n")
    else:
        if(str(tree) == '0'):
            color_property = "style=filled fillcolor=\"darksalmon\""
        else:
            color_property = "style=filled fillcolor=\"darkolivegreen1\""
        dot_file.write("\t"+node_name +" [ " +color_property +"label=\"" +str(tree).upper() +"\" ]\n")
        node_counter += 1
    return node_name

def dicretize_data(df):
    # CREDIT
    quantiles = [.1, .25, .50, .75, .9]
    credit_thresholds = [0]
    for value in df[Ex1_Headers.CREDIT_AMOUNT.value].quantile(quantiles):
        credit_thresholds.append(int(value))
    new_header = 'aux'
    #credit_thresholds sería = [0, 934, 1365, 2319, 3972, 7179]
    credit_thresholds = [0, 1000, 1500, 2500, 4000, 7000]
    for i in range(len(credit_thresholds)-1):
        df.loc[df[Ex1_Headers.CREDIT_AMOUNT.value].between(credit_thresholds[i], credit_thresholds[i+1], inclusive="left"),
            new_header] = '$'+str(credit_thresholds[i])+"-"+str(credit_thresholds[i+1])
    df.loc[df[Ex1_Headers.CREDIT_AMOUNT.value] >= credit_thresholds[len(credit_thresholds)-1], new_header] = 'Mayor a $'+str(credit_thresholds[len(credit_thresholds)-1])
    df = df.drop(columns=[Ex1_Headers.CREDIT_AMOUNT.value]).rename(columns={new_header: Ex1_Headers.CREDIT_AMOUNT.value})

    # AGES
    age_ceilings = [0, 25, 40, 65]
    for i in range(len(age_ceilings)-1):
        df.loc[df[Ex1_Headers.AGE.value].between(age_ceilings[i], age_ceilings[i+1], inclusive="left"),
            new_header] = str(age_ceilings[i])+"-"+str(age_ceilings[i+1])
    df.loc[df[Ex1_Headers.AGE.value] >= age_ceilings[len(age_ceilings)-1], new_header] = str(age_ceilings[len(age_ceilings)-1])+"+"
    df = df.drop(columns=[Ex1_Headers.AGE.value]).rename(columns={new_header: Ex1_Headers.AGE.value})

    # LENGTH
    quantiles = [.1, .25, .50, .75, .9]
    duration_ceilings = [0]
    for value in df[Ex1_Headers.CREDIT_DURATION.value].quantile(quantiles):
        duration_ceilings.append(int(value))
    for i in range(len(duration_ceilings)-1):
        df.loc[df[Ex1_Headers.CREDIT_DURATION.value].between(duration_ceilings[i], duration_ceilings[i+1], inclusive="left"),
            new_header] = str(duration_ceilings[i])+"-"+str(duration_ceilings[i+1])+" m"
    df.loc[df[Ex1_Headers.CREDIT_DURATION.value] >= duration_ceilings[len(duration_ceilings)-1], new_header] = str(duration_ceilings[len(duration_ceilings)-1])+'+ m'
    df = df.drop(columns=[Ex1_Headers.CREDIT_DURATION.value]).rename(columns={new_header: Ex1_Headers.CREDIT_DURATION.value})
    return df

def perform_classification(train, test, tree, goal_attribute):
    total_train_elements, total_test_elements = train.shape[0], test.shape[0]
    possible_answers = train[goal_attribute].unique()

    error = 0
    confusion = np.zeros((len(possible_answers), len(possible_answers)))

    for i in range(total_test_elements):
        entry = test.iloc[i]
        actual = entry[goal_attribute]
        prediction = classify(entry, tree)
        confusion[actual][prediction] += 1
        if actual != prediction:
            error += 1
    precision = get_precision(confusion)
    accuracy = get_accuracy(confusion)
    print('---------------------------')
    print('Error --> ', error, '\nAccuracy --> ',
          accuracy, '\nPrecision --> ', precision)
    print('---------------------------')
    if Configuration.isVerbose():
        plot_confusion_matrix(confusion, ["REJECTED", "APPROVED"])
    return error, accuracy, precision

def classify(entry, tree):
    if type(tree) is not dict:
        return tree
    attribute_to_check = tree['attribute']
    entry_value = entry[attribute_to_check]
    branch_to_explore = tree['children'][entry_value]
    return classify(entry, branch_to_explore)

def count_tree_nodes(tree):
    if type(tree) is not dict:
        return 1
    sum = 1
    for child in tree['children'].keys():
        sum += count_tree_nodes(tree['children'][child])
    return sum

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
            print("Started building tree at", datetime.datetime.now())
            tree = make_tree(df, train, Ex1_Headers.CREDITABILITY.value)
            print("Finished building tree at", datetime.datetime.now())
            export_tree(tree)
            perform_classification(train, test, tree, Ex1_Headers.CREDITABILITY.value)
        else:
            run_cross_validation(df=df, cross_k=cross_validation_k)
