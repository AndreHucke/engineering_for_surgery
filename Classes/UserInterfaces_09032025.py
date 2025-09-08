import numpy as np
from sklearn import datasets
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from PCA import *
from myBoxplot import *
from matplotlib.widgets import Button
from matplotlib.widgets import TextBox

# load Fisher's Iris dataset
iris = datasets.load_iris()
meas = iris.data
species = iris.target_names
species_num = iris.target
sz = np.shape(meas)

p = pca(meas)
D_pca = p.project(meas)

# demo of callbacks in an interactive figure
class interactiveFigure:
    def __init__(self, D):
        self.fig, self.ax = plt.subplots()
        #connect callbacks
        plt.connect('motion_notify_event', self.on_mouse_move)
        plt.connect('button_press_event', self.on_mouse_click)
        plt.connect('key_press_event', self.on_key_press)

        # add a plot
        plt.scatter(D[:,0], D[:,1])

        # turn interactive mode on
        plt.ion()

        # show() does not block in interactive mode
        plt.show()

        # loop while the figure exists re-drawing it and processing events
        while plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(.3)

    # prints mouse coordinates to screen
    def on_mouse_move(self, event):
        if event.inaxes:
            print(f'x={event.xdata:.3f}, y={event.ydata:.3f}')

    # print button clicked to screen
    def on_mouse_click(self,event):
        if event.button is MouseButton.LEFT:
            print('Left click')
        elif event.button is MouseButton.RIGHT:
            print('Right click')

    # print keyboard press to screen
    def on_key_press(self, event):
        print(event.key)


# create an interactive visualization tool to compare samples across scatter/box plots
class interactiveIrisFigure:
    def __init__(self,meas, species, species_num, D_pca):
        # save data as member attributs
        self.D = meas
        self.labels = species
        self.y = species_num
        self.D_pca = D_pca

        # create a figure with axes for plots, button, text boxes
        self.fig = plt.figure()
        self.ax = [plt.axes([.15,.275,.3,.7]),
                   plt.axes([.65,.275,.3,.7]),
                   plt.axes([0.075, 0.075, .1, .1]),
                   plt.axes([.5, 0.075, .1, .1]),
                   plt.axes([0.8, 0.075, 0.1, 0.1])]
        # plt.connect('motion_notify_event',self.on_mouse_move)
        plt.connect('button_press_event',self.on_mouse_click)
        # plt.connect('key_press_event',self.on_key_press)

        self.b1 = Button(self.ax[2], 'Update')
        self.f1 = 1
        self.f2 = 2
        self.text_box1 = TextBox(self.ax[3], 'x-axis PCA feat.',initial=f'{self.f1}')
        self.text_box2 = TextBox(self.ax[4],'y-axis PCA feat.',initial=f'{self.f2}')
        self.b1.on_clicked(self.on_button_press)


        plt.ion()

        self.redraw()

        while plt.fignum_exists(self.fig.number):
            self.fig.canvas.draw_idle()
            self.fig.canvas.start_event_loop(.3)

    # function to redraw base scatter/boxplots
    def redraw(self):
        plt.axes(self.ax[0])
        plt.cla()
        for i in range(len(self.labels)):
            plt.scatter(self.D_pca[np.nonzero(self.y==i),self.f1-1],
                        self.D_pca[np.nonzero(self.y==i),self.f2-1],
                        label=self.labels[i])
        plt.legend()
        plt.xlabel(f'PCA Feature {self.f1}')
        plt.ylabel(f'PCA Feature {self.f2}')
        self.ax[0].axis('equal')

        plt.axes(self.ax[1])
        plt.cla()
        myBoxplot(self.D)
        plt.xlabel('Raw features')
        plt.ylabel('Feature values')

        plt.show()

    # function to update pca features selected for scatter plot
    def on_button_press(self, event):
        try:
            f1 = np.fromstring(self.text_box1.text, dtype=int, sep=' ')[0]
            if f1 < 1:
                f1=1
            if f1 > np.shape(self.D_pca)[0]:
                f1 = np.shape(self.D_pca)[0]
        except:
            f1 = 1

        try:
            f2 = np.fromstring(self.text_box2.text, dtype=int, sep=' ')[0]
            if f2 < 1:
                f2=1
            if f2 > np.shape(self.D_pca)[0]:
                f2 = np.shape(self.D_pca)[0]
        except:
            f2 = 1

        if f1 == f2:
            f2 = (f1 %(np.shape(self.D_pca)[0])) + 1

        self.text_box1.set_val(f'{f1}')
        self.text_box2.set_val(f'{f2}')
        self.f1 = f1
        self.f2 = f2
        self.redraw()

    def on_mouse_click(self,event):
        if event.button is MouseButton.LEFT and event.inaxes == self.ax[0]:
            # find closest datapoint to cursor position where mouse click occured
            x = event.xdata
            y = event.ydata
            f1 = self.f1-1
            f2 = self.f2-1
            dsq = (self.D_pca[:,f1] - x)**2 + (self.D_pca[:,f2] - y)**2
            i = np.argmin(dsq)

            self.redraw()
            plt.axes(self.ax[0])

            # plot thick black star over selected datapoint
            plt.plot(self.D_pca[i,f1], self.D_pca[i,f2], 'k*', linewidth=3)

            # add text that indicates the sample number and corresponding raw feature value
            # display it below if the data is in the upper half of the plot, otherwise above
            if self.D_pca[i,f2] > 0:
                plt.text(self.ax[0].dataLim.x0, self.D_pca[i,f2] - .75*(
                    self.ax[0].dataLim.y1 - self.ax[0].dataLim.y0
                ), f'Sample {i+1}\nRaw feature value {self.D[i,0]:.1f} '
                   f'{self.D[i,1]:.1f} {self.D[i,2]:.1f} {self.D[i,3]:.1f}')
            else:
                plt.text(self.ax[0].dataLim.x0,self.D_pca[i,f2] + .5 * (
                        self.ax[0].dataLim.y1 - self.ax[0].dataLim.y0
                ),f'Sample {i + 1}\nRaw feature value {self.D[i,0]:.1f} '
                  f'{self.D[i,1]:.1f} {self.D[i,2]:.1f} {self.D[i,3]:.1f}')


            # use black circles to indicate the same sample in the boxplots
            plt.axes(self.ax[1])
            plt.plot([1,2,3,4], self.D[i,:], 'ko')


if __name__ == '__main__':
    i = interactiveFigure(D_pca)
    f = interactiveIrisFigure(meas, species, species_num, D_pca)