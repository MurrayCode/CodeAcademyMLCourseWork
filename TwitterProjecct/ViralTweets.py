import pandas as pd
import numpy as np 
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


all_tweets = pd.read_json("random_tweets.json", lines=True)

#exploring data
print(all_tweets.loc[0]['user']['location'])
print(all_tweets['retweet_count'])
#get median value for retweets
median_retweet = all_tweets["retweet_count"].median()
print(median_retweet)
#create is_viral column and add tweets with a retweet value > median
all_tweets['is_viral'] = np.where(all_tweets['retweet_count'] > 
median_retweet,1,0)

print(all_tweets['is_viral'].value_counts())
#create feature columns, tweet length, follower_count, friends_count, num_hashtags
all_tweets['tweet_length'] = all_tweets.apply(lambda tweet: len(tweet['text']), axis=1)
all_tweets['followers_count'] = all_tweets.apply(lambda tweet: tweet['user']['followers_count'], axis=1)
all_tweets['friends_count'] = all_tweets.apply(lambda tweet: tweet['user']['friends_count'], axis=1)
all_tweets['num_hashtags'] = all_tweets.apply(lambda tweet: tweet['text'].count('#'), axis = 1)

#print(all_tweets['followers_count'])
#print(all_tweets['num_hashtags'])
#Create features and labels for training
labels = all_tweets['is_viral']
data = all_tweets[['tweet_length', 'followers_count','friends_count','num_hashtags']]

#normailse the data in features
scaled_data = scale(data)
print(scaled_data[0])
#split the data
train_data, test_data, train_labels, test_labels = train_test_split(scaled_data, labels, test_size = 0.2, random_state = 1)

#loop through values 1 - 200 to find best value for K
scores = []
for k in range(1,200):
    test_classifier = KNeighborsClassifier(n_neighbors=k)
    test_classifier.fit(train_data, train_labels)
    scores.append(test_classifier.score(test_data, test_labels))
#plot on graph K values
x_axis = range(1,200)
plt.plot(x_axis, scores)
plt.show()

#create classifier with best K value
final_classifier = KNeighborsClassifier(n_neighbors=45)
final_classifier.fit(train_data, train_labels)
print(final_classifier.score(test_data, test_labels))
