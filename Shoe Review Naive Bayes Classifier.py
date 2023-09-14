import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

data = pd.read_csv("shoe_reviews_1.txt", delimiter = ":")

def basic_split(data, y, length, split_point = 0.5): #split data 50-50
    n = int(split_point * length)
    X_train = data[:n].copy() #All data points till the split point
    X_test = data[n:].copy() #all points after split
    y_train = y[:n].copy()
    y_test = y[n:].copy()
    return X_train, X_test, y_train, y_test