import logistic
from config.best_separation import BestSeparation
from config.constants import ALL_HEADERS
from fileHandling import read_csv, export_csv, print_entire_df
import numpy as np
import pandas as pd
from datetime import datetime

FILE = './input/acath_good.csv'
ITERATIONS = 50

if __name__ == '__main__':
    # Marking the run as storing the best config
    BestSeparation.setActive(True)
    df = read_csv(FILE)
    for cross_k in [5, 10, 15, 20]:
        for i in range(ITERATIONS):
            if i == 0:
                data = np.array([logistic.run_logistic(df, cross_k=cross_k, account_male_female=True)])
            else:
                data = np.append(data, [logistic.run_logistic(df, cross_k=cross_k, account_male_female=True)], axis = 0)
        # Create dataframe with data
        result_df = pd.DataFrame(data, columns = ['Accuracy','Accuracy stderr','Precision','Precision stderr'])
        # Export dataframe
        export_csv(result_df, f'./results/metrics_k_{cross_k}_iters_{ITERATIONS}_{int(datetime.timestamp(datetime.now()))}.csv', delimiter=';')
    train_df, test_df = BestSeparation.getDfs()
    train_df = pd.DataFrame(train_df[:, 0:6], columns = ALL_HEADERS)
    test_df = pd.DataFrame(test_df[:, 0:6], columns = ALL_HEADERS)
    ts = int(datetime.timestamp(datetime.now()))
    export_csv(train_df, f'./results/best_train_{BestSeparation.getBestAccuracy()}_k_{BestSeparation.getCrossK()}_{ts}.csv', delimiter=';')
    export_csv(test_df, f'./results/best_test_{BestSeparation.getBestAccuracy()}_k_{BestSeparation.getCrossK()}_{ts}.csv', delimiter=';')
    