import codecademylib3_seaborn
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

#load the dataset breast cancer into a variable 
breast_cancer_data = load_breast_cancer()
#prints the data, and targets
print(breast_cancer_data.data[0])
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)

#splits the data into training features and labels & validation/testing features and labels
training_data, validation_data, training_labels, validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size = 0.2, random_state = 100)
print(len(training_data))     
print(len(training_labels))
#initilize new list to store each accuracy score
accuracies = []

#loops through 1 - 100 and tries them for n_neighbors to find best accuracy - stores these in accuracy list
for k in range(1,101):
  classifier = KNeighborsClassifier(n_neighbors =k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data, validation_labels))

#plots the accuracies onto a graph
k_list = [*range(1,101)]
plt.plot(k_list, accuracies)
plt.xlabel("k")
plt.ylabel("Validation Accuracy")
plt.show()
