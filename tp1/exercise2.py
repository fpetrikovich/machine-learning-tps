import numpy as np
import pandas as pd
from bayes import apply_bayes
from fileHandling import read_data
from newsProcessing import get_key_words, compute_laplace_frequencies, compute_class_probability, build_binary_survey
from constants import Ex2_Mode, Ex2_Headers, Ex2_Categoria
from configurations import Configuration

def run_exercise_2(file, mode):
    print('Importing news data...')
    df = read_data(file)

    allowed_categories = [Ex2_Categoria.DEPORTES, Ex2_Categoria.SALUD, Ex2_Categoria.ENTRETENIMIENTO, Ex2_Categoria.ECONOMIA]
    allowed_categories = [e.value for e in allowed_categories]
    df = df[df[Ex2_Headers.CATEGORIA.value].isin(allowed_categories)]
    msk = np.random.rand(len(df)) < 0.9
    train = df[msk]
    test = df[~msk]

    print('Processing training set...')
    key_words = get_key_words(train, allowed_categories, 50, is_analysis=mode == Ex2_Mode.ANALYZE.value)
    if mode == Ex2_Mode.SOLVE.value:
        frequencies = compute_laplace_frequencies(train, key_words, allowed_categories)
        class_probability = compute_class_probability(train, allowed_categories)

        print('Processing testing set...')
        test_df = build_binary_survey(test, key_words)
        total_elements, current_step = test_df.shape[0], 0
        for index in range(total_elements):
            if index / 25 > current_step:
                current_step += 1
                print('Processing', current_step * 25, 'out of', total_elements)
            # Print the headline for a better reference
            if Configuration.isVerbose():
                print("\n------------------------\n")
                print(test.iloc[index][Ex2_Headers.TITULAR.value], '-->', test.iloc[index][Ex2_Headers.CATEGORIA.value])
                print('')
            # Use as a sample the indexed location
            apply_bayes(test_df.iloc[[index]].reset_index(drop=True), frequencies, class_probability, key_words, Ex2_Headers.CATEGORIA.value, allowed_categories, print_example=Configuration.isVeryVerbose())
