import matplotlib.pyplot as plt
import random
import time
import numpy as np

# Make a animation of a sine wave
# def animate_sine_wave():
#     num_points = 100
#     x = np.linspace(0, 2 * np.pi, num_points)
#     y = np.sin(x)
#     fig, ax = plt.subplots()
#     line, = ax.plot(x, y)
#     ax.set_ylim(-1.5, 1.5)
#     ax.set_xlim(0, 2 * np.pi)
#     ax.set_xlabel('x')
#     ax.set_ylabel('sin(x)')
#     ax.set_title('Sine Wave Animation')
#     for i in range(100):
#         print(i)
#         y = np.sin(x + i / 10.0)
#         line.set_ydata(y)
#         plt.draw()
#         plt.pause(0.000001)
#     plt.show()

def animate_sine_wave2():
    num_points = 200
    x = np.linspace(0, 4 * np.pi, num_points)
    y1 = np.sin(x)
    y2 = np.cos(x)
    fig, ax = plt.subplots()
    line, = ax.plot(x[0:100], y1[0:100], label='sin(x)')
    line2, = ax.plot(x[0:100], y2[0:100], label='cos(x)')
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xlabel('x')
    ax.set_ylabel('sin(x)')
    ax.set_title('Sine Wave Animation')
    
    for i in range(100):
        print(i)
        y1 = np.sin(x[i:i + 100])
        y2 = np.cos(x[i:i + 100])
        line.set_ydata(y1)
        line2.set_ydata(y2)
        plt.draw()
        plt.pause(0.00001)
        # plt.pause(0.000001)
    
    plt.show()


def animate_sine_wave3():
    num_points = 200
    x = np.linspace(0, 4 * np.pi, num_points)
    y1 = np.sin(x)
    y2 = np.cos(x)
    fig, ax = plt.subplots()
    line, = ax.plot(x[0:100], y1[0:100], label='sin(x)')
    line2, = ax.plot(x[0:100], y2[0:100], label='cos(x)')
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlim(0, 2 * np.pi)
    ax.set_xlabel('x')
    ax.set_ylabel('sin(x)')
    ax.set_title('Sine Wave Animation')
    
    for i in range(100):
        print(i)
        y1 = np.sin(x[i:i + 100])
        y2 = np.cos(x[i:i + 100])
        line.set_ydata(y1)
        line2.set_ydata(y2)
        plt.draw()
        plt.pause(0.00001)
        # plt.pause(0.000001)
    
    plt.show()

if __name__ == "__main__":
    # Call the function to animate the sine wave
    animate_sine_wave3()
    
