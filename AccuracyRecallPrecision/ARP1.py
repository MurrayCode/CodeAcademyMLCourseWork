labels = [1, 0, 0, 1, 1, 1, 0, 1, 1, 1]
guesses = [0, 1, 1, 1, 1, 0, 1, 0, 1, 0]

true_positives = 0 
true_negatives = 0
false_positives = 0
false_negatives = 0

for i in range(len(guesses)):
  if guesses[i] == 1 and labels[i] == 1:
    true_positives +=1
  elif guesses[i] == 0 and labels[i] == 0:
    true_negatives +=1
  elif guesses[i] == 1 and labels[i] == 0:
    false_positives +=1
  elif guesses[i] == 0 and labels [i] == 1:
    false_negatives +=1

accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
