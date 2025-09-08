import numpy as np
import matplotlib.pyplot as plt

from PCA import pca

D = np.array([[-5, 2.5], [2, -1], [10, -5], [-12, 6]])

fig, ax = plt.subplots()
plt.plot(D[:,0], D[:,1], 'o')
plt.xlim(-15, 15)
plt.ylim(-15, 15)

# Rebase for a 1 dimension linear space
e = np.array([[-2, 1]])

print(2.5*e)
print(-1*e)
print(-5*e)
print(6*e)

a = np.array([[2.5, -1, 5, 6]]).T

# @ symbol is a matrix multiplication
print(a @ e)

c = plt.get_cmap('plasma')
an = np.linspace(-10, 10, len(c.colors))
for i in range(len(an)):
    plt.plot(an[i] * e[0,0], an[i] * e[0,1], 'o', color=c.colors[i], markersize=0.5)

p = pca(D)
print(p.evects[:,0])
print(p.evals)

import json
import numpy as np
import matplotlib.pyplot as plt

f = open ('project_1\humact.json', 'rt')

d = json.load(f)
f.close()

feat = np.array(d['feat'])
print(f'JSON: {np.shape(feat)}')

plt.show()

class myPca(pca):
    def __init__(self, d):
        super().__init__(d)

    def num_effective_dims(self, perc_var_thrsh):

        result = 50
        return result

p = myPca(D)

D_pca = p.project(D)
print(p.num_effective_dims(99.9))