from sklearn.feature_extraction.text import CountVectorizer
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re
from sklearn.svm import SVC


train_set = []

with open("train.csv") as training_file:
    training_file.readline()
    for line in training_file:
        x, y = line.strip().split(",")
        train_set.append((x, y))

X, y = zip(*train_set)
y = np.array(y)

vectorizer = CountVectorizer(analyzer="char_wb", ngram_range=(2, 6))

# Get n-grams for analization
X = vectorizer.fit_transform(X).astype("f")

labels = ["BE", "BS", "LU", "ZH"]
n_gram_sums = np.squeeze(np.array(
    [np.sum(X[y == label], axis=0) for label in labels]))

probabilities = []

for idx in range(len(labels)):
    sum_all = np.sum(n_gram_sums, axis=0)
    probability = n_gram_sums[idx] / sum_all
    probabilities.append(probability)

probabilities = np.array(probabilities)

alpha = 0.1
min_probability_in_dialects = np.min(probabilities, axis=0)
max_probability_in_dialects = np.max(probabilities, axis=0)

max_condition = max_probability_in_dialects >= 0.35
min_condition = min_probability_in_dialects <= 0.15

mask = np.argwhere(np.logical_or(min_condition, max_condition))
mask = np.squeeze(mask)

resulting_features = X[:, mask]


def plot():
    means = []
    for i in range(10000):
        means.append(np.mean(np.random.choice(probabilities.flatten(), 100)))
    means = np.array(means)
    #means = (means - means.mean()) / means.std()
    sns.distplot(probabilities.flatten(), bins=30)
    plt.show()


plot()
