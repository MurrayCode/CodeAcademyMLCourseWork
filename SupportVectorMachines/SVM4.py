import codecademylib3_seaborn
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from graph import points, labels, draw_points, draw_margin

classifier = SVC(kernel='linear', C = 0.51)
points.append([3,3])
points.append([4,4])
points.append([5,5])
labels.append(0)
labels.append(1)
labels.append(0)
classifier.fit(points, labels)

draw_points(points, labels)
draw_margin(classifier)

plt.show()
