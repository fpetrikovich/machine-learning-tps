# Get row for specific class
from fileHandling import print_entire_df
from configurations import Configuration
from constants import Memory_Keys

def get_df_for_class(df, header_name, _class):
    return df[df[header_name] == _class].reset_index(drop=True)

def prefilter_dfs(frequencies, class_probability, class_header, classes, header_names):
    # Create the map
    memory = {}
    memory[Memory_Keys.FREQUENCIES] = {}
    memory[Memory_Keys.PROBABILITY] = {}
    memory[Memory_Keys.CLASS_PROBABILITY] = {}
    memory[Memory_Keys.KEY_FREQUENCIES] = {}
    for _class in classes:
        memory[Memory_Keys.FREQUENCIES][_class] = get_df_for_class(frequencies, class_header, _class)
        memory[Memory_Keys.KEY_FREQUENCIES][_class] = {}
        for header in header_names:
            memory[Memory_Keys.KEY_FREQUENCIES][_class][header] = memory[Memory_Keys.FREQUENCIES][_class].at[0,header]
        memory[Memory_Keys.PROBABILITY][_class] = get_df_for_class(class_probability, class_header, _class)
        memory[Memory_Keys.CLASS_PROBABILITY][_class] = memory[Memory_Keys.PROBABILITY][_class].iat[0,0]
    # Lose reference for garbage collector
    memory[Memory_Keys.PROBABILITY] = 0
    memory[Memory_Keys.FREQUENCIES] = 0
    return memory

def compute_hmap_for_class(example_memory, header_names, class_to_calculate, memory):
    # Start with the probability of the class itself
    probability = memory[Memory_Keys.CLASS_PROBABILITY][class_to_calculate]
    for header in header_names:
        # Determine if we need the probability of it being a "positive" or a "negative" example
        # If it's a positive example, just multiply, otherwise use 1 - probability
        if (example_memory[header] == 1):
            probability *= memory[Memory_Keys.KEY_FREQUENCIES][class_to_calculate][header]
        else:
            probability *= (1 - memory[Memory_Keys.KEY_FREQUENCIES][class_to_calculate][header])
    return probability

def apply_bayes(example, memory, header_names, class_header, class_names, print_example = True):
    results, total, max, max_classification = {}, 0, 0, None
    # Maps the probabilities so that it can access way faster
    example_memory = {header:example[header][0] for header in header_names}
    # Calculate the hmap without denominator
    for classification in class_names:
        # Compute the result for each nationality
        results[classification] = compute_hmap_for_class(example_memory, header_names, classification, memory)
        # Add it towards the total so that we get the correct probability
        total += results[classification]
    # Pretty prints
    if print_example:
        print("Current example to classify...")
        print_entire_df(example)
        print('')
    # Iterate again to properly compute the probability
    for classification in class_names:
        # Divide by the total
        results[classification] = results[classification] / total
        if Configuration.isVerbose():
            print("Probability of the example to be", classification, "is", results[classification])
        # Check which one is the maximum one
        if results[classification] > max:
            max = results[classification]
            max_classification = classification
    if Configuration.isVerbose():
        print("It's most likely to be", max_classification)
    return max_classification, example[class_header][0], results
