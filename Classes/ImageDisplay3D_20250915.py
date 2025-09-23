import matplotlib.pyplot as plt
import numpy as np
import json

# Load a CT image to demo 3D display
f = open('Classes/CT.json', encoding='utf-8-sig')
d = json.load(f)
f.close()

img = np.array(d['data'], dtype=np.int16)
voxsz = np.array(d['voxsz'], dtype=np.float64)

print(np.shape(img))
print(voxsz)

# Choose a 2D slice to visualize, start with Axial slice 80, custom level+contrast
slc = 80
level = 0 # gray
contrast = 1000 # -500 and below will be black, +500 and above will be white
# CT uses Hounsfield Units typically limitted to range [-1024, 3071]
# this level+contrast maps white to intensities >=500, black to <=-500, gray to 0

# first axial direction
fig, ax = plt.subplots()
ax.imshow(img[:,:,slc].T, 'gray', interpolation='bilinear',
          vmin=level-contrast/2, vmax=level+contrast/2)
ax.set_aspect(voxsz[1]/voxsz[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Slice z = {slc}')

# repeat with sagittal slice direction
slc = 255
fig, ax = plt.subplots()
ax.imshow(img[slc,:,:].T, 'gray', interpolation='bilinear',
          vmin=level-contrast/2, vmax=level+contrast/2)
ax.set_aspect(voxsz[2]/voxsz[1])
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
plt.xlabel('y')
plt.ylabel('z')
plt.title(f'Slice x = {slc}')

# coronal direction
slc = 255
fig, ax = plt.subplots()
ax.imshow(img[:,slc,:].T, 'gray', interpolation='nearest',
          vmin=level-contrast/2, vmax=level+contrast/2)
ax.set_aspect(voxsz[2]/voxsz[0])
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
plt.xlabel('x')
plt.ylabel('z')
plt.title(f'Slice y = {slc}')

# Try the interactive volumeViewer class I am providing
from volumeViewer import *
vv = volumeViewer()
vv.setImage(img, voxsz, contrast=contrast, level=level)

msk = img > 1000
vv.addMask(msk, color=[0,1,0])

# controls:
# main window:
#   up,down,'a','z' page through slices
#   'g','v' adjust contrast
#   'd','c' adjust level
#   escape or q to close the figure
#   double left-click centers all three views on a point
#   double right-click resets the view
#   Pyplot's built in zoom/pan functions
# 3D window:
#   hold right click to zoom
#   hold left click to rotate
#   hold middle-mouse button to pan
#   press 'u' to pick a point
vv.display()

plt.show()
