import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.DESCR)
print(digits.data)
print(digits.target)

plt.gray()
plt.matshow(digits.images[100])
plt.show()

print(digits.target[100])

model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)

fig = plt.figure(figsize=(8,3))
fig.suptitle('Cluser Center Images', fontsize = 14, fontweight='bold')

for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)

plt.show()

new_samples = np.array([

[0.00,0.68,3.58,2.35,0.07,0.00,0.00,0.00,0.00,1.97,7.54,7.62,5.69,0.60,0.00,0.00,0.00,1.67,7.39,6.54,7.46,7.15,4.10,0.30,0.00,5.39,6.92,0.66,1.51,5.17,7.54,4.78,0.00,6.32,5.25,0.00,0.00,0.53,6.92,5.55,0.00,6.46,6.00,1.06,3.26,6.69,7.46,2.10,0.00,2.95,7.62,7.62,7.62,6.23,1.58,0.00,0.00,0.00,1.67,2.29,1.36,0.00,0.00,0.00],

[0.00,0.00,0.38,0.76,0.76,0.00,0.00,0.00,0.22,5.38,7.62,7.62,7.62,6.08,1.65,0.00,2.12,7.62,4.47,2.36,3.50,7.62,6.46,0.00,0.83,7.46,6.31,3.50,4.03,7.62,6.69,0.00,0.00,2.26,6.47,7.62,7.54,7.54,4.70,0.00,0.00,0.00,0.00,0.00,0.00,6.85,4.56,0.00,0.00,0.00,0.00,0.00,0.00,6.78,4.56,0.00,0.00,0.00,0.00,0.00,0.00,2.05,1.14,0.00],

[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.45,7.23,3.04,0.00,0.00,0.00,0.00,0.00,0.00,6.92,5.32,0.00,0.00,0.00,0.00,0.00,0.00,4.33,7.31,0.15,0.00,0.00,0.00,0.00,0.00,2.58,7.62,1.67,0.00,0.00,0.00,0.00,0.00,1.58,7.62,2.27,0.00,0.00,0.00,0.00,0.00,1.51,7.62,2.21,0.00,0.00,0.00,0.00,0.00,0.08,2.82,0.30,0.00,0.00],

[0.00,0.00,0.00,0.07,3.57,2.36,0.00,0.00,0.00,0.00,0.00,4.62,7.62,3.71,0.00,0.00,0.00,0.00,2.73,7.62,4.10,0.00,0.00,0.00,0.00,0.45,7.00,5.93,0.08,0.00,0.00,0.00,0.00,4.63,7.62,7.62,7.08,6.70,3.34,0.00,0.90,7.54,6.99,7.47,7.62,7.62,6.23,0.00,0.75,7.07,7.62,7.16,5.78,4.41,1.06,0.00,0.00,0.23,0.76,0.08,0.00,0.00,0.00,0.00]

])

new_labels = model.predict(new_samples)
print(new_labels)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')
