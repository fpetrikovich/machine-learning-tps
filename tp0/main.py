import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from constants import Headers, Modes
import matplotlib.gridspec as gridspec

def print_entire_df (df):
    pd.set_option('display.max_rows', df.shape[0]+1)
    print(df)

def read_data(file):
    df = pd.read_excel(file)
    return df

def process_replace(df):
    return df.replace(to_replace=999.99, value=np.nan)

# Filter all the N/A rows
# Replace all 999.99 values in the column with the calculation for that column
def process_mean(df):
    for header in Headers:
        # Not process Sexo
        if header != Headers.SEXO:
            h = header.value
            filter = df[h] != 999.99
            df[h] = df[h].replace(to_replace=999.99, value=np.mean(df[filter][h]))
    return df

def process_median(df):
    for header in Headers:
        if header != Headers.SEXO:
            h = header.value
            filter = df[h] != 999.99
            df[h] = df[h].replace(to_replace=999.99, value=np.median(df[filter][h]))
    return df

# Processing changes according to given mode
def process_data(df, mode):
    if mode == Modes.REMOVE.name:
        df = process_replace(df)
    elif mode == Modes.MEDIAN.name:
        df = process_mean(df)
    elif mode == Modes.MEAN.name:
        df = process_median(df)
    return df

def create_sex_boxplots(input_df, field):
    df = input_df[input_df[field].notnull()]
    men = df[df[Headers.SEXO.value] == 'M'][field]
    women = df[df[Headers.SEXO.value] == 'F'][field]
    data = [df[field], men, women]
    position_values = [0, 1, 2]
    plt.boxplot(data, positions = position_values)
    plt.xticks(position_values[::], ["Global", "Hombres", "Mujeres"])
    plt.title("Consumo de " +field +" por sexo")
    plt.show()

def create_alcohol_graphs(input_df):
    alcohol = Headers.ALCOHOL.value
    calories = Headers.CALORIAS.value
    df = input_df[(input_df[calories].notnull()) & (input_df[alcohol].notnull())]
    cate1 = df[df[calories] <= 1100]
    cate2 = df[(df[calories] > 1100) & (df[calories] <= 1700)]
    cate3 = df[df[calories] > 1700]
    data = [cate1[alcohol], cate2[alcohol], cate3[alcohol]]
    position_values = [0, 1, 2]
    plt.boxplot(data, positions = position_values)
    plt.xticks(position_values[::], ["CATE1", "CATE2", "CATE3"])
    plt.title("Consumo de alcohol por categoría calórica")
    plt.show()
    # Create scattterplot
    plt.scatter(cate1[calories], cate1[alcohol])
    plt.scatter(cate2[calories], cate2[alcohol])
    plt.scatter(cate3[calories], cate3[alcohol])
    plt.xlabel("Consumo calórico")
    plt.ylabel("Consumo de alcohol")
    plt.show()

def create_sex_histograms(input_df):

    n_bins = 10
    colors = ['gray', 'blue', 'pink']
    labels = ['All', 'Men', 'Women']

    # Graph one figure with the 3 histograms
    fig, ((ax0), (ax1), (ax2)) = plt.subplots(nrows=3, ncols=1)

    graphs = {
        Headers.GRASAS.value: ax0, 
        Headers.ALCOHOL.value: ax1, 
        Headers.CALORIAS.value: ax2
    }

    for header in Headers:
        if header != Headers.SEXO:
            h = header.value
            graph = graphs[h] 

            # Filter data of header by sex
            df = input_df[input_df[h].notnull()]
            men = df[df[Headers.SEXO.value] == 'M'][h]
            women = df[df[Headers.SEXO.value] == 'F'][h]
            data = [df[h], men, women]

            # plot all, men, and women data in the histogram
            graph.hist(data, n_bins, histtype='bar', color=colors, label=labels)
            graph.legend(prop={'size': 8})
            graph.set_title(h)            

    fig.tight_layout()
    plt.savefig('graphs/Figure_6_Histogram.png')

def covariance(input_df):
    calories_cond = input_df[Headers.CALORIAS.value].notnull() 
    alcohol_cond = input_df[Headers.ALCOHOL.value].notnull()
    sat_fat_cond = input_df[Headers.GRASAS.value].notnull()
    
    df = input_df[calories_cond & alcohol_cond & sat_fat_cond]

    fat = df[Headers.GRASAS.value].to_numpy()
    alcohol = df[Headers.ALCOHOL.value].to_numpy()
    calories = df[Headers.CALORIAS.value].to_numpy()

    data = np.stack([fat, alcohol, calories])

    correlation_matrix = np.corrcoef(data)
    print (correlation_matrix)
    return correlation_matrix


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Machine Learning TP0")

    # Add arguments
    parser.add_argument('-f', dest='file', required=True)
    parser.add_argument('-m', dest='mode', required=True)
    args = parser.parse_args()

    df = read_data(args.file)
    new_df = process_data(df, args.mode)
    print_entire_df(new_df)

    # Create BoxPlot and Scatterplot graphs
    for header in Headers:
        if header != Headers.SEXO:
            h = header.value
            create_sex_boxplots(new_df, header.value)
    
    create_alcohol_graphs(new_df)
    create_sex_histograms(new_df)

    covariance(new_df)

if __name__ == '__main__':
    main()
