import pandas as pd

def read_data(file):
    df = pd.read_excel(file)
    return df