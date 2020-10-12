import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
# From sklearn.cluster, import KMeans class

iris = datasets.load_iris()

samples = iris.data

# Use KMeans() to create a model that finds 3 clusters
model = KMeans(n_clusters = 3)
# Use .fit() to fit the model to samples
model.fit(samples)
# Use .predict() to determine the labels of samples 
print(model.predict(samples))

