from fileHandling import read_csv, df_to_numpy
from kmedias import KMeans
from config.constants import Headers

def run_KMeans(file, k):
    df = read_csv(file)
    df = df[[Headers.AGE.value, Headers.CHOLESTEROL.value]]
    df_np = df_to_numpy(df)
    print(df_np)

    kmeans = KMeans(df_np, k)
    point, centroide = kmeans.apply()


def plot_2d_example():
    a = 4