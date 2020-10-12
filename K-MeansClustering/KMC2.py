import codecademylib3_seaborn
import matplotlib.pyplot as plt

from sklearn import datasets

iris = datasets.load_iris()

# Store iris.data
samples = iris.data
# Create x and y
x = [samples[0]]
y = [samples[1]]
# Plot x and y
plt.scatter(x,y, alpha = 0.5)
# Show the plot
plt.show()
