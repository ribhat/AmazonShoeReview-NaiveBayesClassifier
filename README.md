# AmazonShoeReview-NaiveBayesClassifier

This Project analyzes different product reviews for a men’s running shoe, and based on word choice, should be able to classify the review as either positive or negative. In order to do this, I first gathered a collection of product reviews from an Amazon product page. The collection of reviews contains an equal number of positive and negative reviews so that they can be trained/tested evenly. Once the data is collected, it is split into train/test data. We will compare our results when using a 50-50 split and an 80-20 split.

The idea behind the classifier is to train a model to classify a review as positive or negative based on the prior probability of their word choice. For example, reviews containing words such as “hate” or “regret” can typically be seen as negative reviews. Thus, by providing our model with enough training data, we can teach it to classify words as generally positive or negative, based on the results it gathered from the training data. If a certain word can be used both positively or negatively, the one with the higher probability is selected.

In order to gather the data, I went to a product page on Amazon, scrolled down to the bottom and clicked on customer comments. From this point, I found reviews in both the positive and negative categories, based on
my own discretion. From here, we were able to create the total bag of words. Now that the data is stored (either in a .txt or .csv), it has to be read into a data frame, using the panda’s library. We want to ensure that the formatting of the strings is similar, since, for example, we don’t want our classifier to think ‘hate’ and ‘Hate’ are two separate words. Once the bag of words is created, the data must be split into training and test. Originally, we will use a 50-50 split and later compare the results to repeating the process with an 80-20 split. First, we must create our training/testing dataset and calculate prior probabilities and conditional independence for each feature in our bag of words. The probability of positive and negative are compared and the larger value is chosen as the classification.


This project was originally completed on Jupyter Notebooks using python and is available to download from the master branch. There is also a .py file with the same contents


