# Get row for specific class
from fileHandling import print_entire_df
from configurations import Configuration

def get_df_for_class(df, header_name, _class):
    return df[df[header_name] == _class]

def compute_hmap_for_class(sample, frequencies, class_probability, header_names, class_header, class_to_calculate):
    # Get the probabilities for just this class
    current_class_frequencies = get_df_for_class(frequencies, class_header, class_to_calculate)
    current_class_probability = get_df_for_class(class_probability, class_header, class_to_calculate)
    # Start with the probability of the class itself
    probability = current_class_probability.iloc[0][0]
    for header in header_names:
        # Ignore class name header (nationality / category)
        if header != class_header:
            # Determine if we need the probability of it being a "positive" or a "negative" example
            # If it's a positive example, just multiply, otherwise use 1 - probability
            if (sample[header][0] == 1):
                probability *= current_class_frequencies.iloc[0][header]
            else:
                probability *= (1 - current_class_frequencies.iloc[0][header])
    return probability

def apply_bayes(example, frequencies, class_probability, header_names, class_header, class_names, print_example = True):
    results, total, max, max_classification = {}, 0, 0, None
    # Calculate the hmap without denominator
    for classification in class_names:
        # Compute the result for each nationality
        results[classification] = compute_hmap_for_class(example, frequencies, class_probability, header_names, class_header, classification)
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
