from PCA import pca
import json
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset humact.json
f = open('Engineering_for_Surgery/project_1/humact.json', 'rt')
dataset = json.load(f)
f.close()

# Take a look at the keys in the json file
print(dataset.keys())

# Looking at the json data, we see that the the possible keys are dict_keys(['Description', 'actid', 'actnames', 'feat', 'featlabels'])
actid = np.array(dataset['actid'])
actnames = np.array(dataset['actnames'])
feat = np.array(dataset['feat'])
featlabels = np.array(dataset['featlabels'])

print(f'JSON: {np.shape(feat)}')
print(f'Activity IDs: {np.unique(actid)}')
print(f'Activity Names: {actnames}')
print(f'Feature Labels: {np.unique(featlabels)}')

# Plot the features in a scatter plot
# r_feat = np.random.randint(0, feat.shape[1], size=2)
fig, ax = plt.subplots(1 , 2, figsize=(15, 5))

# Plot a random pair of features using the labels
ax[0].set_title(f'Features {0} and {1}')
# Add activity labels to the scatter plot
scatter = ax[0].scatter(feat[:, 0], feat[:, 1], c=actid, cmap='OrRd')
handles, labels = scatter.legend_elements()
ax[0].legend(handles, actnames, title="Activity Names", loc="best")

# Perform PCA on the features
class myPca(pca):
    def __init__(self, d):
        super().__init__(d)

p = myPca(feat)

D_pca = p.project(feat)

# Plot the first two principal components in a scatter plot with activity labels and names on the legend
scatter = ax[1].scatter(D_pca[:, 0], D_pca[:, 1], c=actid, cmap='OrRd')
ax[1].set_title('First two principal components')
handles, labels = scatter.legend_elements()
ax[1].legend(handles, actnames, title="Activity Names", loc="best")

# Compute persaon correlation coefficient raw feature 0 and raw feaute 1
corr = np.corrcoef(feat[:, 0], feat[:, 1])
# Print the rho value
print(f'Correlation coefficient between feature 0 and feature 1: {corr[0, 1]}')

# Computer the persaon correlation coefficient PCA feature 0 and PCA feature 1
corr_pca = np.corrcoef(D_pca[:, 0], D_pca[:, 1])
# Print the rho value
print(f'Correlation coefficient between PCA feature 0 and PCA feature 1: {corr_pca[0, 1]}')

# Now let's see the number of effective dimensions to explain 99.9% of the variance
n_d = p.num_effective_dims(99.9)
print(f'Number of effective dimensions to explain 99.9% of the variance: {n_d}')

plt.show()