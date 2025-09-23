from PCA import pca
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset DimensionalityReduction.json
f = open('Engineering_for_Surgery/project_1/DimensionalityReduction.json', 'rt')
dataset = json.load(f)
f.close()

feat = np.array(dataset)
print(f'JSON: {np.shape(feat)}')

# Plot the data in a scatter plot to understand that we can't see the hidden message (pick a random pair of features)
r_feat = np.random.randint(0, feat.shape[1], size=2)
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].plot(feat[:, r_feat[0]], feat[:, r_feat[1]], 'o')
ax[0].set_title(f'Features {r_feat[0]} and {r_feat[1]}')

# Do the dimensionality reduction by first defining a pca, coming from the provided PCA class
class myPca(pca):
    def __init__(self, d):
        super().__init__(d)

p = myPca(feat)

# Project the data into the PCA space
D_pca = p.project(feat)

# Plot the first two principal components in a scatter plot to see the message: Message says exp(j*pi)+1=0
ax[1].plot(D_pca[:, 0], D_pca[:, 1], 'o')
plt.show()