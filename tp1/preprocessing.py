from constants import Ex2_Blacklist, Ex2_Categoria, Ex2_Headers
from functools import reduce

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
        print("Category", category.value, "\n")
        current_map = mapping[category]
        i = 0
        for key in current_map:
            if i > n: break
            print(key, "-->", current_map[key])
            i += 1
        print("\n-------------------\n")

def preprocess_news(df):
    mapping = {}
    for category in Ex2_Categoria:
        # DF with specific category
        filter = df[Ex2_Headers.CATEGORIA.value] == category.value
        # Get a slice of the DF
        filtered_df = df[filter]
        # Map to the tokenized versions
        filtered_df.loc[filter, Ex2_Headers.TITULAR.value] = list(map(lambda x: tokenize_title(x), filtered_df[Ex2_Headers.TITULAR.value]))
        # Store the result
        mapping[category] = reduce(accumulate_map_in_map, filtered_df[Ex2_Headers.TITULAR.value], {})
        # Sort the map
        mapping[category] = dict(sorted(mapping[category].items(), key=lambda item: item[1], reverse=True))
    # Show a top 20 by category
    show_top_n_by_category(mapping, 30)
    
