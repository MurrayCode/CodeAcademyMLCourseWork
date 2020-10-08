#sets a list of labels & guesses to simulate a ML algorithm
labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

#initilize variables
true_positives = 0 
true_negatives = 0
false_positives = 0
false_negatives = 0

#runs through each guess 
#if guess and lablel are both 1 add 1 to true positive
#if they are both 0 add one to true negative 
#if guess is 1 and label is 0 add 1 to false positive
#if guess is 0 and label is 1 add one to false negative
for i in range(len(guesses)):
  if guesses[i] == 1 and labels[i] == 1:
    true_positives +=1
  elif guesses[i] == 0 and labels[i] == 0:
    true_negatives +=1
  elif guesses[i] == 1 and labels[i] == 0:
    false_positives +=1
  elif guesses[i] == 0 and labels [i] == 1:
    false_negatives +=1
    
#accuracy is calculated using the formula A = (TP + TN) / (TP + TN +_FP + FN) or A = (TP + TN) / (length of guesses)
accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
accuracy = (true_positives + true_negatives) / len(guesses)
#recall is calculated by R = TP /(TP + FN)
recall = true_positives / (true_positives +false_negatives)
#prevision is calculated by P = tp / (TP + FP)
precision = true_positives / (true_positives + false_positives)
#f1 score is calculated by F1 = (P * R) / (P + R)
f_1 = 2 * (precision * recall) / (precision + recall)
print(f_1)
print(precision)