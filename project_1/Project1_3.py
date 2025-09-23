# % Script for Project 1 part 3
# % ECE 5370: Engineering for Surgery
# % Fall 2024
# % Author: Prof. Jack Noble; jack.noble@vanderbilt.edu

import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.backend_bases import MouseButton
import adjustText as aT

# loading the dataset
f = open('Engineering_for_Surgery/project_1/EEG Eye State.json','rt')
dataset = json.load(f)
f.close()
data = np.array(dataset['data'])

# filter outliers
rng=1.5
for i in range(0,np.shape(data)[1]):
    sts = np.quantile(data[:,i],[.01,.5,.99])
    arr = data[:,i] > sts[1] + rng*(sts[2]-sts[1])
    data[arr,i] = sts[1] + rng*(sts[2]-sts[1])
    arr = data[:, i] < sts[1] + rng * (sts[0] - sts[1])
    data[arr, i] = sts[1] + rng * (sts[0] - sts[1])

# trim to 2000 timepoints
data = data[0:1999,:]

# initialize plots
f, ax = plt.subplots(2,2)
fnum = plt.get_fignums()[-1]

#define a class that will update the base plots and in which you will define your mouse button callback code
class Project1:
    def __init__(self, data, ax):
        self.data = data
        self.ax = ax
        plt.ion()

    def UpdatePlot(self):
        plt.axes(ax[0][0])
        plt.cla()
        plt.plot(data[:, 0])
        plt.plot(data[:, np.shape(data)[1] - 1] * (np.max(data[:, 0]) - np.min(data[:, 0])) + np.min(data[:, 0]), '--')
        plt.title('Sensor 1')
        plt.axes(ax[0][1])
        plt.cla()
        plt.plot(data[:, 1])
        plt.plot(data[:, np.shape(data)[1] - 1] * (np.max(data[:, 1]) - np.min(data[:, 1])) + np.min(data[:, 1]), '--')
        plt.title('Sensor 2')
        plt.axes(ax[1][0])
        plt.cla()
        plt.plot(data[:, 2])
        plt.plot(data[:, np.shape(data)[1] - 1] * (np.max(data[:, 2]) - np.min(data[:, 2])) + np.min(data[:, 2]), '--')
        plt.title('Sensor 3')
        plt.axes(ax[1][1])
        plt.cla()
        h1 = ax[1][1].plot(data[:, 0], label='Sensor_1')
        h2 = ax[1][1].plot(data[:, 1], label='Sensor_2')
        h3 = ax[1][1].plot(data[:, 2], label='Sensor_3')
        ax[1][1].legend()

    def on_mouse_click(self, event):
        # First, if the click was made whithin the first 3 axis ([0,0], [0,1], [1,0]), then:
        # 1- Determine which timepoint the click was made (closest point using eigendistace)
        # 2- Do nothing if the click does not fall whithin a valid timepoint
        # 3- If the timepoint is valid:
        # 3.1- Displat a red star at the correspoding timepoint in all 3 axis
        # 3.2- Display the value of that timepoint in all 3 axis as well as the datapoint with 2 decimals points
        # 3.3- Clear the plot and redraw the plot prior to displaying the star and text

        # Second, if the click was made within the last axis ([1,1]), then:
        # 1- Determine which timepoint the click was made (closest point using eigendistace)
        # 2- Do nothing if the click does not fall whithin a valid timepoint
        # 3- If the timepoint is valid:
        # 3.1- Display a color star for each one of the curves in this subplot with the height equals to the height of the curve for that feature
        # 3.2- Display the data above the star
        # 3.3- Clear the plot and redraw the plot prior to displaying the star and text

        if event.button is MouseButton.LEFT:
            if event.inaxes in [ax[0][0], ax[0][1], ax[1][0]]:
                tp = int(np.round(event.xdata))
                if tp >= 0 and tp < np.shape(data)[0]:
                    self.UpdatePlot()
                    for i in range(3):
                        self.ax[i // 2][i % 2].plot(tp, data[tp, i], 'r*', markersize=10)
                        # Position text above the star
                        y_offset = (self.ax[i // 2][i % 2].get_ylim()[1] - self.ax[i // 2][i % 2].get_ylim()[0]) * 0.05
                        self.ax[i // 2][i % 2].text(tp, data[tp, i] + y_offset, f'tp:{tp}\nval:{data[tp, i]:.2f}', 
                                                   color='red', fontsize=10, ha='center', weight='bold',
                                                   bbox=dict(facecolor='white', alpha=0.6, edgecolor='red'))
                    plt.draw()
            elif event.inaxes == ax[1][1]:
                tp = int(np.round(event.xdata))
                if tp >= 0 and tp < np.shape(data)[0]:
                    self.UpdatePlot()
                    colors = ['blue', 'orange', 'green']
                    for i in range(3):
                        self.ax[1][1].plot(tp, data[tp, i], '*', color=colors[i], markersize=10)
                        y_offset = (self.ax[1][1].get_ylim()[1] - self.ax[1][1].get_ylim()[0]) * 0.05
                        self.ax[1][1].text(tp, data[tp, i] + y_offset, f'tp:{tp}\nval:{data[tp, i]:.2f}', 
                                          color=colors[i], fontsize=12, ha='center', weight='bold',
                                          bbox=dict(facecolor='white', alpha=0.6, edgecolor=colors[i]))
                    aT.adjust_text(self.ax[1][1].texts, only_move={'points':'y', 'text':'y'},
                                   arrowprops=dict(arrowstyle='->', color=colors[i]))
                    plt.draw()

if __name__=='__main__':
    # initiate a Project1 object
    p = Project1(data, ax)
    p.UpdatePlot()

    # Need an infinite while-loop to use the figure interactively
    plt.ion()
    cid = f.canvas.mpl_connect('button_press_event', p.on_mouse_click)
    plt.show()
    while 1:
        if plt.fignum_exists(fnum) == False:
            break
        f.canvas.draw_idle()
        f.canvas.start_event_loop(0.3)
