import codecademylib3_seaborn
from LinearRegression3 import gradient_descent
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("heights.csv")

X = df["height"]
y = df["weight"]
b = 0
m = 0
b, m = gradient_descent(X, y, num_iterations=1000, learning_rate=0.0001)
y_predictions = [m*x + b for x in X]
plt.plot(X,y_predictions, y, 'o')
#plot your line here:
plt.show()


