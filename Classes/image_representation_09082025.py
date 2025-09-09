import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

# Small example from scratch
img = np.zeros((10,15))
for i in range(10):
    for j in range(15):
        img[i,j] = i * j

# Visualize the image
fig, ax  = plt.subplots()
plt.imshow(img, cmap='gray')
plt.colorbar()
plt.xlabel('columns')
plt.ylabel('rows')

# Load a tif image
img2 = cv.imread('Engineering_for_Surgery/Classes/cameraman.tif')

# Print the shape of the image
print(np.shape(img2)) # This shows that the image is 3D, but we don't need the third channel for gray scale
img2 = img2[:,0:250,0] # Keep only one channel, and trimming the width to 250 pixels

# Plot the image
fig, ax = plt.subplots()
ax.imshow(img2, cmap='gray')

# Covert all the pixel in the cameraman's face to  gray
img2_anon = img2.copy()
img2_anon[40:86, 90:136] = 127

fig, ax = plt.subplots()
ax.imshow(img2_anon, cmap='gray')

# Save as tif the anonimazed image
cv.imwrite('Engeneering_for_Surgery/Classes/cameraman_anon.tif', img2_anon)

# Side by side image and histogram to see the pixel distribution
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
cols = ax[0].imshow(img2, 'gray')
plt.colorbar(cols)
plt.axes(ax[1])
plt.hist(img2.ravel(), bins = np.linspace(0,255,256))
plt.yscale('log')
plt.ylabel('# of occurrences')
plt.xlabel("Pixel intensities")

# Segment using Otsu's method
thresh, img2_otsu = cv.threshold(img2, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU) # cv has a built-in Otsu's method!
plt.plot([thresh, thresh], [0, 3000], 'r--', label='Otsu threshold')
fig2, ax2 = plt.subplots()
ax2.imshow(img2_otsu, 'gray')

# Doing isoconturing
X, Y = np.meshgrid([0,1], [2,3,4])
print(X)
print(Y)

# Close for contouring example
plt.close(fig2)

C, R = np.meshgrid(np.arange(np.shape(img2)[1]), np.arange(np.shape(img2)[0]))
plt.figure(fig)
plt.axes(ax[0])
plt.contour(C, R, img2_otsu, levels=[127.5], colors='r')
plt.contour(C, R, img2, levels=[thresh + 0.5], colors='b') # Add 0.5 so the boundary is between the pixels

# Create a interactive class to select the threshold
class imageThresholdSelector:
    def __init__(self, fig, ax, img):
        self.fig = fig
        self.ax = ax
        self.img = img
        self.thresh = 0

        self.mn = np.amin(self.img)
        self.mx = np.amax(self.img)

        self.C, self.R = np.meshgrid(np.arange(np.shape(self.img)[1]), np.arange(np.shape(self.img)[0]))
        plt.connect('button_press_event', self.on_mouse_click)

        plt.ion()
        plt.show()

        while (1):
            fig.canvas.draw_idle()
            fig.canvas.start_event_loop(0.1)

    def on_mouse_click(self, event):
        if event.inaxes != self.ax[0]:
            self.thresh = event.xdata
            if self.thresh >= self.mn and self.thresh < self.mx:
                plt.axes(self.ax[0])
                plt.cla()
                self.ax[0].imshow(self.img, 'gray')
                plt.contour(self.C, self.R, self.img, levels=[self.thresh + 0.5], colors='r')

                plt.axes(self.ax[1])
                plt.cla()
                plt.hist(self.img.ravel(), bins = np.linspace(0,255,256))
                plt.yscale('log')
                plt.ylabel('# of occurrences')
                plt.xlabel("Pixel intensities")

                plt.plot([self.thresh, self.thresh], [0, 3000], 'r--', label='Selected threshold')

i = imageThresholdSelector(fig, ax, img2)
plt.show()