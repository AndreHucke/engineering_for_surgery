import matplotlib.pyplot as plt
import numpy as np
import json

f = open('Engineering_for_Surgery/Classes/CT.json', encoding='utf-8-sig')
d = json.load(f)
f.close()

img = np.array(d['data'], dtype=np.int16)
voxsz = np.array(d['voxsz'], dtype=np.float64)

print(np.shape(img))
print(voxsz)

slc = 80
level = 0
contrast = 1000
# CT uses Hounsfield units tipically limited to range [-1024, 3071]

fig, ax = plt.subplots()
ax.imshow(img[:, :, slc].T, cmap='gray', interpolation='bilinear',
          vmin=level - contrast / 2, vmax=level + contrast / 2)
ax.set_aspect(voxsz[1] / voxsz[0])
plt.xlabel('x')
plt.ylabel('y')
plt.title(f"Slice z = {slc}")

slc = 255
fig, ax = plt.subplots()
ax.imshow(img[slc, :, :].T, cmap='gray', interpolation='bilinear',
          vmin=level - contrast / 2, vmax=level + contrast / 2)
ax.set_aspect(voxsz[2] / voxsz[1])
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
plt.xlabel('y')
plt.ylabel('z')
plt.title(f"Slice x = {slc}")

slc = 255
fig, ax = plt.subplots()
ax.imshow(img[:, slc, :].T, cmap='gray', interpolation='nearest',
          vmin=level - contrast / 2, vmax=level + contrast / 2)
ax.set_aspect(voxsz[2] / voxsz[0])
ax.set_ylim(bottom=0, top=np.shape(img)[2]-1)
plt.xlabel('x')
plt.ylabel('z')
plt.title(f"Slice y = {slc}")

plt.show()
