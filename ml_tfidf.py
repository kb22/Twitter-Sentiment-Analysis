import numpy as np
import pandas as pd
from termcolor import colored
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

# Load data
print(colored("Loading train and test data", "yellow"))
train_data = pd.read_csv('data/clean_train.csv')
test_data = pd.read_csv('data/clean_test.csv')
print(colored("Data loaded", "yellow"))

# Tf-IDF
print(colored("Applying TF-IDF transformation", "yellow"))
tfidfVectorizer = TfidfVectorizer(min_df = 5, max_features = 1000)
tfidfVectorizer.fit(train_data['Clean_tweet'].apply(lambda x: np.str_(x)))
train_tweet_vector = tfidfVectorizer.transform(train_data['Clean_tweet'].apply(lambda x: np.str_(x)))
test_tweet_vector = tfidfVectorizer.transform(test_data['Clean_tweet'].apply(lambda x: np.str_(x)))

# Training
print(colored("Training Random Forest Classifier", "yellow"))
randomForestClassifier = RandomForestClassifier()
randomForestClassifier.fit(train_tweet_vector, train_data['Sentiment'])

# Prediction
print(colored("Predicting on train data", "yellow"))
prediction = randomForestClassifier.predict(train_tweet_vector)
print(colored("Training accuracy: {}%".format(accuracy_score(train_data['Sentiment'], prediction)*100), "green"))

print(colored("Predicting on test data", "yellow"))
prediction = randomForestClassifier.predict(test_tweet_vector)
print(colored("Testing accuracy: {}%".format(accuracy_score(test_data['Sentiment'], prediction)*100), "green"))