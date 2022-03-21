from fileHandling import read_data
from preprocessing import preprocess_news
from constants import Ex2_Mode

def run_exercise_2(file, mode):
    print('Importing news data...')
    df = read_data(file)

    if mode == Ex2_Mode.ANALYZE.value:
        print('Processing news...')
        preprocess_news(df)