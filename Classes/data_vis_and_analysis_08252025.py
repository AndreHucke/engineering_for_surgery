import numpy as np
from sklearn import datasets

# Load iris dataset and extract features and target
iris = datasets.load_iris()
meas = iris.data
species = iris.target_names
species_num = iris.target

# Looking at the shapes of the extracted data
sz = np.shape(meas)
# print(sz)
# print(species)
# print(species[species_num])

# print(f'Species: {species[0]} \t Features: {meas[0,0]:.1f}, {meas[0,1]:.1f}, {meas[0,2]:.1f}, {meas[0,3]:.1f}')

# print(f'Species: {species[0]} \t Features: {meas[0,0]:5.1f}, {meas[0,1]:5.1f}, {meas[0,2]:5.1f}, {meas[0,3]:5.1f}')

# # Print all the measurements in the dataset
# for i in range(sz[0]):
#     print(f'{i+1} \t Species: {species[species_num[i]]} \t Features: {meas[i,0]:5.1f} {meas[i,1]:5.1f} {meas[i,2]:5.1f} {meas[i,3]:5.1f}')

import matplotlib.pyplot as plt

# Plotting the dataset as line and dots
# fig, ax = plt.subplots()
# ax.plot(meas[0:50, 0])
# ax.plot(meas[0:50, 0], '.')

# Line and dot plots are kinda garbage for this dataset, so we can use a histogram
[h, x] = np.histogram(meas[0:50, 0])

# Plotting the histogram
# plt.hist(meas[0:50, 0])
# plt.title('Histogram of feature 1 for setosa')
# plt.xlabel('Feature values')
# plt.ylabel('# of occurrences')

# Plot feature 2
# fig, ax = plt.subplots()
# plt.hist(meas[0:50, 1])
# plt.title('Histogram of feature 2 for setosa')
# plt.xlabel('Feature values')
# plt.ylabel('# of occurrences')

# Careful with the axis. They are different ranges. To change this:
mn = np.amin(meas) - .1
mx = np.amax(meas) + .1
bns = np.linspace(mn, mx, 21) # Corresponds to 20 bins

# fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)

# ax[0, 0].hist(meas[0:50, 0], bins=bns)
# ax[0, 0].set_title('Feature 1')

# ax[0, 1].hist(meas[0:50, 1], bins=bns)
# ax[0, 1].set_title('Feature 2')

# ax[1, 0].hist(meas[0:50, 2], bins=bns)
# ax[1, 0].set_title('Feature 3')

# ax[1, 1].hist(meas[0:50, 3], bins=bns)
# ax[1, 1].set_title('Feature 4')

# Show all histograms
for i in range(3):
    str = f'Species: {species[i]}'
    plt.figure(str)
    fig, ax = plt.subplots(2, 2, num=str, sharex=True, sharey=True)
    plt.xlim([0,8])
    plt.ylim([0,50])
    for j in range(4):
        ax[j//2, j%2].set_title(f'Feature {j+1}')
        ax[j//2, j%2].hist(meas[50*i:50*i+50, j], bins=bns)

# Scatter plot to visualize the relationship between features
fig, ax = plt.subplots()
plt.scatter(meas[0:50, 0], meas[0:50, 1], marker='.', color=[1,0,0], label=species[0])
plt.scatter(meas[50:100, 0], meas[50:100, 1], marker='x', color=[0,1,0], label=species[1])
plt.scatter(meas[100:150, 0], meas[100:150, 1], marker='s', color=[0,0,1], label=species[2])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Petal Length vs Width')
plt.legend()

# 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(meas[0:50, 0], meas[0:50, 1], meas[0:50, 2], marker='.', color=[1,0,0], label=species[0])
ax.scatter(meas[50:100, 0], meas[50:100, 1], meas[50:100, 2], marker='x', color=[0,1,0], label=species[1])
ax.scatter(meas[100:150, 0], meas[100:150, 1], meas[100:150, 2], marker='s', color=[0,0,1], label=species[2])
ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_zlabel('Sepal Length')
ax.set_title('3D Scatter Plot')
ax.legend()

# For loop on the 2D plots
arr = np.array([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
fig, ax = plt.subplots(2, 3, sharex=True, sharey=True)
for i in range(6):
    cax = ax[i//3, i%3]
    plt.axes(cax)
    plt.scatter(meas[0:50, arr[i,0]], meas[0:50, arr[i,1]], marker='.', color=[1,0,0], label=species[0])
    plt.scatter(meas[50:100, arr[i,0]], meas[50:100, arr[i,1]], marker='x', color=[0,1,0], label=species[1])
    plt.scatter(meas[100:150, arr[i,0]], meas[100:150, arr[i,1]], marker='s', color=[0,0,1], label=species[2])
    
    r = np.corrcoef(meas[:, arr[i,0]], meas[:, arr[i,1]])[0,1]

    plt.xlabel(f'Feature {arr[i,0]+1}; r={r:.2f}')
    plt.ylabel(f'Feature {arr[i,1]+1}')

ax[0,0].legend()

plt.show()