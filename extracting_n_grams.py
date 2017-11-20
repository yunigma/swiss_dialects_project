#!/usr/bin/env python
# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.decomposition import PCA
from collections import defaultdict


# sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
# sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
# sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)
# train_file = open("train.csv", "r")
# train_data = train_file.read()
train_data = pd.read_csv("train.csv")

data_matrix = train_data.as_matrix()

data_BE = []
data_BS = []
data_LU = []
data_ZU = []
for line in data_matrix:
    if line[1] == "BE":
        data_BE.append(line)
    elif line[1] == "BS":
        data_BS.append(line)
    elif line[1] == "LU":
        data_LU.append(line)
    elif line[1] == "ZU":
        data_ZU.append(line)


def exract_n_grams(data):
    """Returns the n-grams contained in @param sentence.
    when the function returns all possible n-grams it takes long... Yet, we can
    just do it once for each of the dialects and save in .txt.
    Or add n as an argument
    """
    n_grams_of_dialect = []
    for sentence in data:
        sentence = sentence[0]
        n = 1
        while n < len(sentence):
            n_grams_of_sentence = list(zip(*[sentence[i:] for i in range(n)]))
            for n_gram in n_grams_of_sentence:
                n_gram = "".join(n_gram)
                n_grams_of_dialect.append(n_gram)
            n += 1

    return n_grams_of_dialect


def count_n_grams(data):
    number_n_grams = defaultdict()
    n_grams = exract_n_grams(data)
    for n_gram in n_grams:
        if n_gram not in number_n_grams:
            number_n_grams[n_gram] = 1
        else:
            number_n_grams[n_gram] += 1

    return number_n_grams


# print(exract_n_grams(data_BE))
# print(count_n_grams(data_BE))
# print(train_data.shape)
# dir(train_data)


# if __name__ == '__main__':
#     main()
