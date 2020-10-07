
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

breast_cancer_data = load_breast_cancer()

print(breast_cancer_data.data[0])
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)

print(len(training_data))     
print(len(training_labels))


classifier = KNeighborsClassifier(n_neighbors =23)
classifier.fit(training_data, training_labels)
print(classifier.score(validation_data, validation_labels))

