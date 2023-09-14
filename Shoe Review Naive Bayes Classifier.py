import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("shoe_reviews_1.txt", delimiter = ":")

def basic_split(data, y, length, split_point = 0.5): #split data 50-50
    n = int(split_point * length)
    X_train = data[:n].copy() #All data points till the split point
    X_test = data[n:].copy() #all poits after split
    y_train = y[:n].copy()
    y_test = y[n:].copy()
    return X_train, X_test, y_train, y_test

vectorizer = CountVectorizer()

X_train, X_test, y_train, y_test = basic_split(data.Review, data.Class, len(data))

X_train = vectorizer.fit_transform(X_train) #fit and transform to training data set
X_test = vectorizer.transform(X_test) #transform for test data set

features = vectorizer.get_feature_names()
print(features)

bag_of_words_first_half = pd.DataFrame(X_train[:,:].todense(), columns = features)
bag_of_words_first_half

class_column_first_half = data.Class[0:6]
class_column_second_half = data.Class[6:]

bag_of_words_first_half['Class'] = class_column_first_half

is_positive = bag_of_words_first_half['Class'] == 'Positive'
positive_df = bag_of_words_first_half[is_positive]
is_negative = bag_of_words_first_half['Class'] == 'Negative'
negative_df = bag_of_words_first_half[is_negative]

positive_prior = 3/6
negative_prior = 3/6
positive_likelihood = []
negative_likelihood = []
for feature in features:
    count = 0
    for item in positive_df[feature]:
        if item != 0:
            count= count + 1 #count number of non zero entries for each word
        positive_likelihood.append(count/len(positive_df))


for feature in features:
    counter = 0
    for item in negative_df[feature]:
        if item != 0:
            counter = counter + 1
        negative_likelihood.append(counter/len(negative_df)) #build negative likelihood list

evidence = []
for feature in features:
    counters = 0
    for item in bag_of_words_first_half[feature]:
        if item != 0:
            counters = counters + 1
    evidence.append(counters/len(bag_of_words_first_half))

positive_posterior = []
negative_posterior = []

for i in range(0, len(features)):
    positive_posterior_i = (positive_likelihood[i] * positive_prior)  # / (evidence[i])
    positive_posterior.append(positive_posterior_i)

for i in range(0, len(positive_posterior)):
    if positive_posterior[i] < 0.00001:
        positive_posterior[i] = .000001


for i in range(0, len(features)):
    negative_posterior_i = (negative_likelihood[i] * negative_prior)#/ (evidence[i])
    negative_posterior.append(negative_posterior_i)
for i in range(0, len(negative_posterior)) :
    if negative_posterior[i] < 0.00001:
        negative_posterior[i] = .000001

bag_of_words_second_half = pd.DataFrame(X_test[:,:].todense(), columns = features)

bag_of_words_second_half['Class'] = ['Negative', 'Positive', 'Negative', 'Positive', 'Negative', 'Positive']

predictions = []
prob_positive = positive_prior
prob_negative = negative_prior
for c in range(0,6):
    index_vector = []
    index = 0
    for item in bag_of_words_second_half.iloc[c]:
        if item != 0 and item != 'Positive' and item != 'Negative':
            index_vector.append(index) #any word that is present in the review, we want to note down the index so we can retrieve it from our features list later
        index = index + 1
    #At this point we have built our index vector for a particular row
    for elem in index_vector: #elem represents features index of every present word in that review
        #word = features[elem]
        #first calculate probability that review is positive
        prob_positive = prob_positive * positive_posterior[elem]
        prob_negative = prob_negative * negative_posterior[elem]
    print(str(prob_positive) + "    " + str(prob_negative))
    if prob_positive >= prob_negative:
        predictions.append('Positive')
    else:
        predictions.append('Negative')

print("predictions: ", predictions)

bag_of_words_second_half['Predicted_Class'] = predictions

#Now we can break an 80-20 split

vectorizer = CountVectorizer()

X_train, X_test, y_train, y_test = basic_split(data.Review, data.Class, len(data), 0.8)

X_train

X_train = vectorizer.fit_transform(X_train) #fit and transform to training data set
X_test = vectorizer.transform(X_test) #transform for test data set

features = vectorizer.get_feature_names()
print(len(features))
print(features)

bag_of_words_train = pd.DataFrame(X_train[:,:].todense(), columns = features)

class_column_train = data.Class[0:9]
class_column_test = data.Class[9:]

bag_of_words_train['Class'] = class_column_train

is_positive = bag_of_words_train['Class'] == 'Positive'
positive_df = bag_of_words_train[is_positive]
is_negative = bag_of_words_train['Class'] == 'Negative'
negative_df = bag_of_words_train[is_negative]

positive_prior = 4/9
negative_prior = 5/9
positive_likelihood = []
negative_likelihood = []
for feature in features:
    count = 0
    for item in positive_df[feature]:
        if item != 0:
            count= count + 1 #count number of non zero entries for each word
        positive_likelihood.append(count/len(positive_df))

for feature in features:
    counter = 0
    for item in negative_df[feature]:
        if item != 0:
            counter = counter + 1
        negative_likelihood.append(counter/len(negative_df)) #build negative likelihood list

positive_posterior = []
negative_posterior = []

for i in range(0, len(features)):
    positive_posterior_i = (positive_likelihood[i] * positive_prior)  # / (evidence[i])
    positive_posterior.append(positive_posterior_i)

for i in range(0, len(positive_posterior)):
    if positive_posterior[i] < 0.00001:
        positive_posterior[i] = .000001

for i in range(0, len(features)):
    negative_posterior_i = (negative_likelihood[i] * negative_prior)#/ (evidence[i])
    negative_posterior.append(negative_posterior_i)
for i in range(0, len(negative_posterior)) :
    if negative_posterior[i] < 0.00001:
        negative_posterior[i] = .000001

bag_of_words_test = pd.DataFrame(X_test[:,:].todense(), columns = features)
bag_of_words_test['Class'] = ['Positive', 'Negative', 'Positive']

predictions = []
prob_positive = positive_prior
prob_negative = negative_prior
for c in range(0,3):
    index_vector = []
    index = 0
    for item in bag_of_words_test.iloc[c]:
        if item != 0 and item != 'Positive' and item != 'Negative':
            index_vector.append(index) #any word that is present in the review, we want to note down the index so we can retrieve it from our features list later
        index = index + 1
    #At this point we have built our index vector for a particular row
    for elem in index_vector: #elem represents features index of every present word in that review
        #word = features[elem]
        #first calculate probability that review is positive
        prob_positive = prob_positive * positive_posterior[elem]
        prob_negative = prob_negative * negative_posterior[elem]
    print(str(prob_positive) + "    " + str(prob_negative))
    if prob_positive >= prob_negative:
        predictions.append('Positive')
    else:
        predictions.append('Negative')

print(predictions)
bag_of_words_test['Predicted_Class'] = predictions