from fileHandling import split_dataset

def run_logistic(df, cross_k = None, account_male_female = False):
    print(df.shape)
    train, test = split_dataset(df, 0.75)
    print(train.shape, test.shape)