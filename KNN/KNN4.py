from movies import movie_dataset, movie_labels

print(movie_dataset['Bruce Almighty'])
print(movie_labels['Bruce Almighty'])

def distance(movie1, movie2):
  squared_difference = 0
  for i in range(len(movie1)):
    squared_difference += (movie1[i] - movie2[i]) ** 2
  final_distance = squared_difference ** 0.5
  return final_distance

def classify(unknown, dataset, k):
  distances = []
  for title in dataset:
    distance_to_point = distance(dataset[title], unknown)
    distances.append([distance_to_point, title])
    distances.sort()
    neighbors = distances[0:k]
  return neighbors

print(classify([.4, .2, .9], movie_dataset, 5))
