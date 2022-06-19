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

def plot_kohonen_colormap_multiple(full_data, separated_data, k=10, colormap='Blues', filename=None, addLabels = True):
    directory = 'results/'
    fig, ((ax1, ax2), (ax3, ax4)) = subplots(2, 2)
    im1 = ax1.pcolormesh(full_data, cmap=colormap, edgecolors=None)
    ax1.set_title("All")
    fig.colorbar(im1, ax=ax1)
    im2 = ax2.pcolormesh(separated_data[0], cmap=colormap, edgecolors=None)
    ax2.set_title("Age")
    fig.colorbar(im2, ax=ax2)
    im3 = ax3.pcolormesh(separated_data[1], cmap=colormap, edgecolors=None)
    ax3.set_title("Car dur")
    fig.colorbar(im3, ax=ax3)
    im4 = ax4.pcolormesh(separated_data[2], cmap=colormap, edgecolors=None)
    ax4.set_title("Cholesterol")
    fig.colorbar(im4, ax=ax4)
    fig.tight_layout()
    # xticks(arange(.5, float(k) + .5), range(k))
    # yticks(arange(.5, float(k) + .5), range(k))

    if filename:
        savefig(directory + filename)
        close()
        print(filename, " color map done!")
    else:
        show()
