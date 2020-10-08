
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

#initilize KNN classifier 
classifier = KNeighborsClassifier(n_neighbors =23)
#fit the training data into classifier
classifier.fit(training_data, training_labels)
#print the accuracy score of the trained classifier against the testing data 
print(classifier.score(validation_data, validation_labels))

