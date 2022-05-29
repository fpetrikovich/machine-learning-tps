import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import argparse

x = np.linspace(0, 5, 10)
y = x

def calculateR2Hiperplanes(weights):
    iterativeHiperplanes = []

    for weight in weights:
        a = float(weight[1])
        b = float(weight[2])
        c = float(weight[0])

        if b != 0: 
            y = -a/b * x - c/b
        else:
            y = 0*x
        iterativeHiperplanes.append(y)

    return iterativeHiperplanes

def graphR2Hiperplane(points, weights):
    hiperplanes = calculateR2Hiperplanes(weights)

    fig, ax = plt.subplots()

    ax.set_xlabel('x', color='#1C2833')
    ax.set_ylabel('y', color='#1C2833')
    ax.grid()

    ax.set_aspect('equal')
    ax.grid(True, which='both')

    # set the x-spine (see below for more info on `set_position`)
    ax.spines['left'].set_position('zero')

    # turn off the right spine/ticks
    ax.spines['right'].set_color('none')
    ax.yaxis.tick_left()

    # set the y-spine
    ax.spines['bottom'].set_position('zero')

    # turn off the top spine/ticks
    ax.spines['top'].set_color('none')
    ax.xaxis.tick_bottom()

    plt.xlim(-0.1,5.1)
    plt.ylim(-0.1,5.1)   

    ax.scatter(points[:,0], points[:,1], c=points[:,2])
    line, = ax.plot(x, hiperplanes[0], '-r', label='Iteration 0')

    def animate(i):
        if (i < len(hiperplanes)):
            line.set_ydata(hiperplanes[i])  # update the data.
        return line,

    ani = animation.FuncAnimation(
        fig, animate, interval=200, blit=False, save_count=50)

    plt.show()
