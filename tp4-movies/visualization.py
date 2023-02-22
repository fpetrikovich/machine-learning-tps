import math
from matplotlib.pyplot import subplots, pcolormesh, colorbar, xticks, yticks, savefig, close, show
from numpy import arange, ndenumerate

def plot_kohonen_colormap(data, k=10, colormap='Blues', filename=None, addLabels = True):
    directory = 'results/'
    fig, ax = subplots(figsize=(k, k))
    pcolormesh(data, cmap=colormap, edgecolors=None)
    colorbar()
    xticks(arange(.5, float(k) + .5), range(k))
    yticks(arange(.5, float(k) + .5), range(k))
    ax.set_aspect('equal')

    if addLabels:
        for (i, j), z in ndenumerate(data):
            ax.text(j + .5 , i + .5, round(z,2), ha='center', va='center', c='r')

    if filename:
        savefig(directory + filename)
        close()
        print(filename, " color map done!")
    else:
        show()

def plot_kohonen_colormap_multiple(full_data, separated_data, labels, k=10, colormap='Blues', filename=None, addLabels = True):
    directory = 'results/'
    cols = math.ceil((len(labels) + 1) / 2.0)
    fig, axes = subplots(2, cols)
    im1 = axes[0][0].pcolormesh(full_data, cmap=colormap, edgecolors=None)
    axes[0][0].set_title("All")
    fig.colorbar(im1, ax=axes[0][0])
    for i in range(len(labels)):
        row = math.floor((i + 1) / cols)
        col = (i + 1) % cols
        im2 = axes[row][col].pcolormesh(separated_data[i], cmap=colormap, edgecolors=None)
        axes[row][col].set_title(labels[i])
        fig.colorbar(im2, ax=axes[row][col])
    fig.tight_layout()
    # xticks(arange(.5, float(k) + .5), range(k))
    # yticks(arange(.5, float(k) + .5), range(k))

    if filename:
        savefig(directory + filename)
        close()
        print(filename, " color map done!")
    else:
        show()
