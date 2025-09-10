import numpy as np
import scipy.ndimage as ndi
from skimage.filters import threshold_multiotsu
from skimage.filters import gaussian
from skimage import morphology
from skimage import measure
import cv2 as cv
import matplotlib.pyplot as plt

# Load a tif image
img = cv.imread('Engineering_for_Surgery/Classes/cameraman.tif')

# Print the shape of the image
print(np.shape(img)) # This shows that the image is 3D, but we don't need the third channel for gray scale
img = img[:,0:250,0] # Keep only one channel, and trimming the width to 250 pixels

# Side by side image and histogram to see the pixel distribution
fig, ax = plt.subplots(1, 2)
plt.axes(ax[0])
cols = ax[0].imshow(img, 'gray')
plt.colorbar(cols)
plt.axes(ax[1])
plt.hist(img.ravel(), bins = np.linspace(0,255,256))
plt.yscale('log')
plt.ylabel('# of occurrences')
plt.xlabel("Pixel intensities")

threshold = threshold_multiotsu(img, classes=2, nbins=256)
print(threshold)
plt.plot([threshold[0], threshold[0]], [0, 10000], 'r')

C, R = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
plt.axes(ax[0])
cntr = plt.contour(C,R,img,levels=[threshold[0]+0.5], colors='red')

sigma=7.4
boxsz = np.round(2+sigma).astype(int)
if (boxsz+1)//2 == boxsz//2:
    boxsz+=1 # Make sure box size is odd.

box = np.ones([boxsz, boxsz])
box /= np.sum(box)

img_box = ndi.convolve(img, box, mode='constant', cval=0)

fig, ax = plt.subplots()
ax.imshow(img_box, cmap='gray')
plt.contour(C, R, img_box, levels=[threshold[0]+0.5], colors='red')

fig, ax = plt.subplots(2, 2)
for i in range(1, 11):
    ax[0, 0].cla()
    ax[0, 0].set_title(f'Sigma={i}')
    ax[0, 1].cla()
    ax[0, 1].set_title(f'Sigma={i}')

    img_box2 = gaussian(img, sigma=i, mode='constant', cval=0) * 255 # have to multiple by 255 because gaussian returns float image in [0,1]

    boxsz = np.round(2+i).astype(int)
    if (boxsz+1)//2 == boxsz//2:
        boxsz+=1 # Make sure box size is odd.

    box = np.ones([boxsz, boxsz])
    box /= np.sum(box)

    img_box = ndi.convolve(img, box, mode='constant', cval=0)

    ax[0, 0].imshow(img_box2, cmap='gray')
    ax[0, 1].imshow(img_box, cmap='gray')
    ax[0, 0].contour(C, R, img_box2, levels=[threshold[0]+0.5], colors='red')
    ax[0, 1].contour(C, R, img_box, levels=[threshold[0]+0.5], colors='red')

    ax[1, 0].cla()
    ax[1, 0].set_title('Gaussian filter over original image')
    ax[1, 0].imshow(img, cmap='gray')
    ax[1, 0].contour(C, R, img_box2, levels=[threshold[0]+0.5], colors='red')

    ax[1, 1].cla()
    ax[1, 1].set_title('Box filter over original image')
    ax[1, 1].imshow(img, cmap='gray')
    ax[1, 1].contour(C, R, img_box, levels=[threshold[0]+0.5], colors='red')

    plt.pause(1)

plt.show()
