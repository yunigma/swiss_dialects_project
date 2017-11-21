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
    CHANGE: Now it returns a list of lists:
    one list has all n_gram of one sentence
    """
    n_grams_of_dialect = []
    for sentence in data:
        sentence = sentence[0]
        n_grams_of_sentence = []
        # n = 1
        # while n < 4:
        #     n_grams_of_sentence = list(zip(*[sentence[i:] for i in range(n)]))
        #     for n_gram in n_grams_of_sentence:
        #         n_gram = "".join(n_gram)
        #         n_grams_of_dialect.append(n_gram)
        #     n += 1
        n = 1
        while n < 4:
            n_grams = list(zip(*[sentence[i:] for i in range(n)]))
            n_grams = ["".join(n_gram) for n_gram in n_grams]
            n_grams_of_sentence.extend(n_grams)
            n += 1

        n_grams_of_dialect.append(n_grams_of_sentence)

    return n_grams_of_dialect


def count_n_grams(data):
    number_n_grams = []
    all_n_grams = exract_n_grams(data)
    for sentence in all_n_grams:
        count_for_sentence = defaultdict()
        for n_gram in sentence:
            if n_gram not in count_for_sentence:
                count_for_sentence[n_gram] = 1
            else:
                count_for_sentence[n_gram] += 1
        number_n_grams.append(count_for_sentence)

    return number_n_grams


# print(exract_n_grams(data_BE))
# print(count_n_grams(data_BE))
# print(train_data.shape)
# dir(train_data)


# if __name__ == '__main__':
#     main()
