import re
import nltk
import numpy as np
import pandas as pd

# Import datasets
train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')

# Function to expand tweet
def expand_tweet(tweet):
	expanded_tweet = []
	for word in tweet:
		if re.search("n't", word):
			expanded_tweet.append(word.split("n't")[0])
			expanded_tweet.append("not")
		else:
			expanded_tweet.append(word)
	return expanded_tweet

# Pre-processing the tweets
print(train_data['Tweet'][0])
train_data['Clean_tweet'] = train_data['Tweet']
train_data['Clean_tweet'] = train_data['Clean_tweet'].str.replace("@[\w]*","")
train_data['Clean_tweet'] = train_data['Clean_tweet'].str.replace("[^a-zA-Z' ]","")
train_data['Clean_tweet'] = train_data['Clean_tweet'].str.split()
train_data['Clean_tweet'] = train_data['Clean_tweet'].apply(lambda tweet: expand_tweet(tweet))
print(train_data['Clean_tweet'][0])
