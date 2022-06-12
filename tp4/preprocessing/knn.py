from fileHandling import scale_df, print_entire_df
import numpy as np
from config.configurations import Configuration

# Hide possible 0/0 warnings
np.seterr(invalid='ignore')


def perform_replacements(df, replacements, target_header):
    # Replaces the missing column data based on the given replacement information
    for replace in replacements:
        _to, _from = int(replace[0]), int(replace[1])
        df.loc[_to, target_header] = df.iloc[_from][target_header]
    return df


def neighbor_accum_fn(accum, curr):
    # Accumulates taking into account the neighbor weight
    if not curr[0] in accum:
        accum[curr[0]] = 0
    accum[curr[0]] += (1 * curr[1])
    return accum


def perform_reduced_classification(train, test, k_neighbors, id_header, calculation_headers, full_headers, print_results=False):
    total_train_elements, total_test_elements = train.shape[0], test.shape[0]
    # Split labels from data
    train_data, test_data = train[calculation_headers].reset_index(
        drop=True), test[calculation_headers].reset_index(drop=True)
    full_train_data, full_test_data = train[full_headers].reset_index(
        drop=True), test[full_headers].reset_index(drop=True)
    # Result
    result = []
    for i in range(total_test_elements):
        distances_with_index = []
        # Get the current example
        example = test_data.iloc[i]
        # Process all differences to get the distance
        example_diff = train_data - example
        # Process the square, then sum it and apply sqrt
        example_diff = ((example_diff**2).sum(axis=1))**.5
        # Add distances & index of example to array
        for j in range(total_train_elements):
            distances_with_index.append((example_diff.iloc[j], j))
        # Sort all items of the array (it will sort by default by ascending distance)
        # Get the nearest k neighbors requested
        sorted_distances_with_index = sorted(
            distances_with_index)[:k_neighbors]
        # Map to classes
        # It maps to tuples like (class, weight), where weight is 1 or 1/distance**2 depending on the mode
        neighbors = list(
            map(lambda x: (full_train_data.iloc[x[1]], x[0]), sorted_distances_with_index))
        if print_results:
            print('----------')
            print("----Point to test----")
            print(full_test_data.iloc[i])
            print("----Neighbors----")
            for n in neighbors:
                print(n)
            print('----------')
        # Just get the first neighbor so that we can replace
        result.append(
            (full_test_data.iloc[i][id_header], neighbors[0][0][id_header]))
    return result


def replace_nearest_neighbor(df, target_header, scaling_headers, id_header, full_headers, calculation_headers):
    # Scale the DF to get proper lengths
    scaled_df = scale_df(
        df, scaling_headers, extra_id_header=id_header)
    if Configuration.isVeryVerbose():
        print_entire_df(scaled_df)
    # Get a filter based on the target header and apply it
    not_null_filter = scaled_df[target_header].notnull()
    train = scaled_df[not_null_filter]
    test = scaled_df[~not_null_filter]
    # Calculate replacements for each of the test elements
    replacements = perform_reduced_classification(train, test, k_neighbors=1, id_header=id_header,
                                                  calculation_headers=calculation_headers, full_headers=full_headers, print_results=Configuration.isVeryVerbose())
    result_df = perform_replacements(df, replacements, target_header=target_header)
    return result_df
