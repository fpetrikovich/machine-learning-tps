from constants import Ex2_Blacklist, Ex2_Headers
from functools import reduce
import pandas as pd

# Accumulator for reducing into a map
def accumulate_in_map(accum, curr):
    if curr in accum:
        accum[curr] += 1
    else:
        accum[curr] = 1
    return accum

# Returns a tokenized map, ignoring the blacklisted words
def tokenize_title(title):
    return reduce(
        accumulate_in_map,
        list(filter(
            lambda part: len(part) >= 4 and not part in Ex2_Blacklist,
            title.split(' ')
            )),
        {}
    )

def accumulate_map_in_map(accum, curr):
    for key in curr:
        if key in accum:
            accum[key] += curr[key]
        else:
            accum[key] = curr[key]
    return accum

def show_top_n_by_category(mapping, n):
    for category in mapping:
        print("Category", category, "\n")
        current_map = mapping[category]
        i = 0
        for key in current_map:
            if i > n: break
            print(key, "-->", current_map[key])
            i += 1
        print("\n-------------------\n")

def build_binary_survey(df, key_words):
    df2 = pd.DataFrame({})
    for index in df.index:
        headline = df[Ex2_Headers.TITULAR.value][index]
        category = df[Ex2_Headers.CATEGORIA.value][index]
        matches = {}
        for word in key_words:
            matches[word] = 0
            if word in headline.split(" "):
                matches[word] = 1
        matches[Ex2_Headers.CATEGORIA.value] = category
        df2 = pd.concat([df2, pd.DataFrame([matches])], ignore_index=True)
    return df2

def compute_laplace_frequencies(df, key_words, possible_categories):
    # Create empty df
    df2 = pd.DataFrame()
    for category in possible_categories:
        matches = {Ex2_Headers.CATEGORIA.value: category}
        for word in key_words:
            matches[word] = 0
        filteredDf = df[df[Ex2_Headers.CATEGORIA.value] == category]
        # Count how many headlines in this category contain each key word
        for index in filteredDf.index:
            headline = filteredDf[Ex2_Headers.TITULAR.value][index]
            category = filteredDf[Ex2_Headers.CATEGORIA.value][index]
            for word in key_words:
                if word in headline.split(" "):
                    matches[word] += 1
        # Apply Laplace and add to df
        for word in key_words:
            matches[word] = (matches[word] + 1) / (filteredDf.shape[0] + len(possible_categories))
        df2 = pd.concat([df2, pd.DataFrame([matches])], ignore_index=True)
    return df2

def compute_class_probability(df, possible_categories):
    _df = pd.DataFrame()
    for category in possible_categories:
        filteredDf = df[df[Ex2_Headers.CATEGORIA.value] == category]
        categoryProbability = pd.DataFrame(data=[[filteredDf.shape[0]/df.shape[0]]])
        categoryProbability[Ex2_Headers.CATEGORIA.value] = category
        _df = pd.concat([_df, categoryProbability], ignore_index = True)
    return _df

def get_mapping(df, possible_categories):
    mapping = {}
    for category in possible_categories:
        # DF with specific category
        filter = df[Ex2_Headers.CATEGORIA.value] == category
        # Get a slice of the DF
        filtered_df = df[filter]
        # Map to the tokenized versions
        filtered_df.loc[filter, Ex2_Headers.TITULAR.value] = list(map(lambda x: tokenize_title(x), filtered_df[Ex2_Headers.TITULAR.value]))
        # Store the result
        mapping[category] = reduce(accumulate_map_in_map, filtered_df[Ex2_Headers.TITULAR.value], {})
        # Sort the map
        mapping[category] = dict(sorted(mapping[category].items(), key=lambda item: item[1], reverse=True))
    return mapping

def get_key_words(df, allowed_categories, n):
    # Count total appearances of words by category
    mapping = get_mapping(df, allowed_categories)
    # Get the top N of each category and group them in a set
    key_words = set({})
    for category in mapping:
        current_map = mapping[category]
        i = 0
        for key in current_map:
            if i > n: break
            key_words.add(key)
            i += 1
    return key_words
