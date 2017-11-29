
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns
import codecs
import random
import unidecode

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
from collections import defaultdict
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier


train_file = codecs.open("train.csv", "r", "UTF-8")
train_file.readline()

# df = pd.read_csv('train.csv')
# print(df.head())
# set(df.Label)
# X = df.Text
# y = df.Label


d = defaultdict(list)
for line in train_file:
    # skip empty lines
    line = line.strip()
    X, y = line.split(",")
    d[y].append(X)

l = []
for k, values in d.items():
    for value in values:
        l.append([value.lower(), k])
# print(l)

random.shuffle(l)
train_X, train_y = zip(*l)

vectorizer = CountVectorizer(
    analyzer="char_wb", ngram_range=(2, 6), encoding=u'utf-8')
classifier = MLPClassifier(
    verbose=True, early_stopping=True, hidden_layer_sizes=(40,))

pipeline = Pipeline([
    ("vectorizer", vectorizer), ("clf", classifier)])

# data = pd.Series(l)
# data = np.asarray(l)
data_BE = [line[0] for line in l if line[1] == "BE"]
data_BS = [line[0] for line in l if line[1] == "BS"]
data_LU = [line[0] for line in l if line[1] == "LU"]
data_ZH = [line[0] for line in l if line[1] == "ZH"]


def get_features(data):
    """ Counts n-grams and prints the frequency histogram
    """
    X = vectorizer.fit_transform(data)
    features = vectorizer.get_feature_names()
    data = X.toarray()
    features_count = data.sum(axis=0)
    dict_features_count = {}

    for f, n in zip(features, features_count):
        dict_features_count[f] = n
    best_features = sorted(dict_features_count.items(), key=lambda x:x[1], reverse=True)[:20]

    n_grams = list(zip(*best_features))[0]
    count = list(zip(*best_features))[1]
    x_pos = np.arange(len(n_grams))

    slope, intercept = np.polyfit(x_pos, count, 1)
    trendline = intercept + (slope * x_pos)

    # plt.plot(x_pos, trendline, color='red', linestyle='--')
    plt.bar(x_pos, count, align='center')
    plt.xticks(x_pos, n_grams)
    plt.ylabel('N-gram Frequency')
    plt.show()


for sentence in data_BE:
    analyze = vectorizer.build_analyzer()
    # print(analyze(sentence))


# get_features(data_BE)
# get_features(data_BS)
# get_features(data_LU)
get_features(data_ZH)
