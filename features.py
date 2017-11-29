
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
    analyzer="char_wb", ngram_range=(2, 3), encoding=u'utf-8')
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
all_dialects = [line[0] for line in l]
# print(all_dialects)


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
    # plt.bar(x_pos, count, align='center')
    # plt.xticks(x_pos, n_grams)
    # plt.ylabel('N-gram Frequency')
    # plt.show()

    return dict_features_count


for sentence in data_BE:
    analyze = vectorizer.build_analyzer()
    # print(analyze(sentence))


features_BE = get_features(data_BE)
features_BS = get_features(data_BS)
features_LU = get_features(data_LU)
features_ZH = get_features(data_ZH)

all_dial_features = get_features(all_dialects)

relative_frequencies = []
for feature in all_dial_features:
    if feature not in features_BE:
        prob1 = 0
    else:
        prob1 = features_BE[feature] / all_dial_features[feature]
    if feature not in features_BS:
        prob2 = 0
    else:
        prob2 = features_BS[feature] / all_dial_features[feature]
    if feature not in features_LU:
        prob3 = 0
    else:
        prob3 = features_LU[feature] / all_dial_features[feature]
    if feature not in features_ZH:
        prob4 = 0
    else:
        prob4 = features_ZH[feature] / all_dial_features[feature]

    relative_frequencies.append((feature, prob1, prob2, prob3, prob4))

# good_features = [f for f in relative_frequencies if 1 in f]
num_features = {}
for f in relative_frequencies:
    num = f[1:]
    num_features[f[0]] = max(num) - 0.25

good_features = sorted(num_features.items(), key=lambda x:x[1], reverse=True)
# good_features = [f[0] for f in sorted(num_features.items(), key=lambda x:x[1], reverse=True)]

# print(good_features)
# print(len(good_features))

item = list(zip(*good_features))[0]
count = list(zip(*good_features))[1]
x_pos = np.arange(len(item))

slope, intercept = np.polyfit(x_pos, count, 1)
trendline = intercept + (slope * x_pos)

plt.plot(x_pos, trendline, color='red', linestyle='--')
plt.bar(x_pos, count, align='center')
plt.xticks(x_pos, item)
plt.ylabel('N-gram Frequency')
plt.show()
