import codecademylib3_seaborn
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()

samples = iris.data

x = samples[:,0]
y = samples[:,1]

# Number of clusters
k = 3
# Create x coordinates of k random centroids
centroids_x = np.random.uniform(min(x), max(x), size=k)
# Create y coordinates of k random centroids
centroids_y = np.random.uniform(min(y), max(y), size=k)

# Create centroids array
centroids = np.array(list(zip(centroids_x, centroids_y)))
# Make a scatter plot of x, y
plt.scatter(y,x)
# Make a scatter plot of the centroids
plt.scatter(centroids_y, centroids_x)
# Display plot
plt.show()
