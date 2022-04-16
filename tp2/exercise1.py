from fileHandling import read_csv, print_entire_df
from constants import Ex1_Headers, EX1_DIVISION, Ex2_Modes, Ex2_Run, Tennis_Headers
from confusion import get_accuracy, get_precision
from plotting import plot_confusion_matrix
from configurations import Configuration
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import datetime
import math
from tree import export_tree

EXAMPLES_UMBRAL = 0
GAIN_UMBRAL = 0
HEIGHT_LIMIT = 2
node_counter = 0

def show_analysis(df):
    df = df.sample(frac=1)
    train_number = int(len(df)/EX1_DIVISION) * (EX1_DIVISION - 1)
    train = df.iloc[0:train_number]
    test = df.iloc[train_number+1:len(df)]
    nodes = []
    train_precisions = []
    test_precisions = []
    examples_u = EXAMPLES_UMBRAL
    gain_u = GAIN_UMBRAL
    max_height = HEIGHT_LIMIT
    for max_height in range(10):
        print("h=", max_height)
        tree = make_tree(df, train, Ex1_Headers.CREDITABILITY.value, examples_u, gain_u, max_height)
        amount_of_nodes = count_tree_nodes(tree)
        nodes.append(amount_of_nodes)
        error, accuracy, precision = perform_classification(train, train, tree, Ex1_Headers.CREDITABILITY.value)
        train_precisions.append(1 - error/train.shape[0])
        error, accuracy, precision = perform_classification(train, test, tree, Ex1_Headers.CREDITABILITY.value)
        test_precisions.append(1 - error/test.shape[0])
    plt.plot(nodes, train_precisions, label = "Training Set Precision", marker='o')
    plt.plot(nodes, test_precisions, label = "Testing Set Precision", marker='o')
    for i in range(len(test_precisions)):
        x = nodes[i]
        y = test_precisions[i]
        if i<2 or (i>1 and x != nodes[i-1]):
            plt.text(x * (1 + 0.01), y * (1 + 0.01), "h="+str(i))
    plt.xlabel('Cantidad de Nodos')
    plt.ylabel('Presición')
    plt.ylim(0.5,1.05)
    plt.legend()
    plt.show()
    # Redo for best one, save the matrix this creates
    best_height = test_precisions.index(max(test_precisions))
    print("Best height was", best_height)
    tree = make_tree(df, train, Ex1_Headers.CREDITABILITY.value, examples_u, gain_u, best_height)
    export_tree(tree)
    perform_classification(train, test, tree, Ex1_Headers.CREDITABILITY.value, True)

def show_analysis_umbrals(df):
    train_number = int(len(df)/EX1_DIVISION) * (EX1_DIVISION - 1)
    examples_u = 0
    gain_u = 0
    max_height = 25
    values = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 1]
    results = []
    for x in values:
        results.append([])
    for i in range(10):
        df = df.sample(frac=1)
        train = df.iloc[0:train_number]
        test = df.iloc[train_number+1:len(df)]
        for index in range(len(values)):
            gain_u = values[index]
            print(examples_u," < examples,", gain_u, " < gain,", "height < ", max_height)
            tree = make_tree(df, train, Ex1_Headers.CREDITABILITY.value, examples_u, gain_u, max_height)
            error, accuracy, precision = perform_classification(train, test, tree, Ex1_Headers.CREDITABILITY.value)
            results[index].append(1 - error/test.shape[0])
    plt.boxplot(results, labels=values)
    plt.xlabel('Umbral de Ganancia')
    plt.ylabel('Presición')
    plt.show()

def run_cross_validation_iteration(i, elements_per_bin, df, goal_attribute, results):
    print('Running cross validation with bin number', i)
    test = df.iloc[i*elements_per_bin:(i+1)*elements_per_bin]
    train = df[~df.index.isin(list(test.index.values))]
    tree = make_tree(df, train, Ex1_Headers.CREDITABILITY.value, EXAMPLES_UMBRAL, GAIN_UMBRAL, HEIGHT_LIMIT)
    error, accuracy, precision = perform_classification(train, test, tree, goal_attribute)
    results[i] = [error, accuracy, precision]

def run_cross_validation(df, cross_k):
    # Calculate number of elements per bin
    elements_per_bin = int(len(df)/cross_k)
    print("Running cross validation using", cross_k,
          "bins with", elements_per_bin, "elements per bin")
    # Iterate and run method
    manager = multiprocessing.Manager()
    # Need this dictionary due to non-shared memory issues
    return_dict = manager.dict()
    jobs = [0] * cross_k
    # Create multiple jobs
    for i in range(cross_k):
        jobs[i] = multiprocessing.Process(target=run_cross_validation_iteration, args=(i, elements_per_bin, df, Ex1_Headers.CREDITABILITY.value, return_dict))
        jobs[i].start()
    # Join the jobs for the results
    for i in range(len(jobs)):
        jobs[i].join()
    # Calculate some metrics
    values = return_dict.values()
    errors = np.array([x[0] for x in values])
    accuracies = np.array([x[1] for x in values])
    print('---------------------------')
    print('---------------------------')
    print('---------------------------')
    print('Error average -->', np.average(errors, axis=0)/elements_per_bin, '\nstd -->', np.std(errors, axis=0))
    print('Accuracy average -->', np.average(accuracies, axis=0), '\nstd -->', np.std(accuracies, axis=0))

'''
This function takes care of building a decision tree
Parameters:
    - df --> Dataframe
    - training_set --> Dataframe with only the training set records
    - goal_attribute --> Column name of the attribute to be decided
    - examples_u --> Pre-poda, number of examples to be required in order to make a decision
    - gain_u --> Pre-poda, gain umbral in order to be taken into account
    - max_height --> Pre-poda, max number of levels to be allowed in the tree
'''
def make_tree(df, training_set, goal_attribute, examples_u, gain_u, max_height):
    map = {}
    for attr in df.columns:
        map[attr] = list(df[attr].unique())
    return ID3(training_set, goal_attribute, map, 0, None, examples_u, gain_u, max_height)

def ID3(df, goal_attribute, attrs_and_values, height, parent_mode, examples_u, gain_u, max_height):
    # STEPS 1-3: Create root
    if df.empty:
        return parent_mode
    goal_df = df[goal_attribute] # Temporarily store the filtered DF to avoid extra instances
    mode = goal_df.mode()[0]
    possible_answers = goal_df.unique()
    if(len(possible_answers) == 1): # Only one answer
        return mode
    # No more information as in you run out of columns to analyze
    if len(df.columns) == 1 or df.shape[0] < examples_u: # No more info, or very few entries
        return mode
    if height > max_height:
        return mode

    # STEP 4: Pick attribute
    gains = {}
    for attr in df.columns:
        if attr != goal_attribute:
            gains[attr] = gain(df, attr, goal_attribute)
    # https://stackoverflow.com/a/280156/10672093
    A = max(gains, key=gains.get)

    # TRIMMING
    if gains[A] < gain_u:
        return mode

    # STEP 4.4:
    '''
    Tree is built recursively using the following structure:
    {
        attribute: ATTR_NAME,
        children: {
            value_for_attr_1: {
                ... repeat
            },
            value_for_attr_2: {
                ... repeat
            },
            ...
        }
    }
    '''
    tree = {'attribute': A, 'children': {}}
    for vi in attrs_and_values[A]:
        # Get entries where it takes value vi, and remove A column
        subtree_df = df[df[A] == vi].loc[:, ~df.columns.isin([A])]
        tree['children'][vi] = ID3(subtree_df, goal_attribute, attrs_and_values, height+1, mode, examples_u, gain_u, max_height)
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
    filtered_df = df[filter_attribute] # Prefilter DF to avoid extra instances
    relative_frequencies = filtered_df.value_counts(normalize=True)
    for v in filtered_df.unique():
        sum += relative_frequencies[v] * HSv(df, goal_attribute, filter_attribute, v)
    return H(df, goal_attribute) - sum

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

def perform_classification(train, test, tree, goal_attribute, show_matrix=False):
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
    if Configuration.isVerbose() or show_matrix:
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

def run_exercise_1(filepath, cross_validation_k=None, mode=Ex2_Run.SOLVE):
    df = read_csv(filepath, ',')
    df = dicretize_data(df)
    # print_entire_df(df)

    if mode == Ex2_Run.ANALYZE:
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
            tree = make_tree(df, train, Ex1_Headers.CREDITABILITY.value, EXAMPLES_UMBRAL, GAIN_UMBRAL, HEIGHT_LIMIT)
            print("Finished building tree at", datetime.datetime.now())
            export_tree(tree)
            perform_classification(train, test, tree, Ex1_Headers.CREDITABILITY.value)
        else:
            run_cross_validation(df=df, cross_k=cross_validation_k)
