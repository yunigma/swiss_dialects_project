#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Mathias Mueller / mathias.mueller@uzh.ch

from __future__ import unicode_literals

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.neural_network import MLPClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import FunctionTransformer, Normalizer
from sklearn.feature_selection import SelectFwe, SelectKBest
from sklearn.feature_selection import chi2, f_classif, mutual_info_classif

from collections import defaultdict

import numpy as np
import pandas as pd
import logging
import argparse
import random
import codecs
import sys
import re
reload(sys)
sys.setdefaultencoding('utf8')

random.seed(42)

sys.stdout = codecs.getwriter('utf-8')(sys.__stdout__)
sys.stderr = codecs.getwriter('utf-8')(sys.__stderr__)
sys.stdin = codecs.getreader('utf-8')(sys.__stdin__)


class CleanTransformer():
    def fit(self, GX, gy):
        # Transform to float for calculations
        X = GX.astype("f")
        y = np.array(gy)

        labels = ["BE", "BS", "LU", "ZH"]

        # Create sums of n-grams - one sum for each
        # feature and dialect
        n_gram_sums = np.squeeze(np.array(
            [np.sum(X[y == label], axis=0) for label in labels]))

        probabilities = []

        # Calculate probabilties of the occurrence of each
        # feature for each dialect
        for idx in range(len(labels)):
            sum_all = np.sum(n_gram_sums, axis=0)
            probability = n_gram_sums[idx] / sum_all
            probabilities.append(probability)

        probabilities = np.array(probabilities)

        min_probability_in_dialects = np.min(probabilities, axis=0)
        max_probability_in_dialects = np.max(probabilities, axis=0)

        # Create conditions which determine if we want to respect
        # the feature in the following classifier
        max_condition = max_probability_in_dialects >= 0.4
        min_condition = min_probability_in_dialects <= 0.15

        # Apply conditions to create a mask which can be used in
        # the transform to filter the feature matrix
        # np.argwhere(np.logical_or(min_condition, max_condition))
        mask = np.argwhere(np.logical_or(min_condition, max_condition))
        self.mask = np.squeeze(mask)

        return self

    def transform(self, X):
        # Return reduced feature set
        return X[:, self.mask]


class Trainer(object):
    """
    Reads raw dialect data and trains a classifier.
    """

    def __init__(self, model="model.pkl", data=None, verbose=False,
                 classifier=None):
        """
        """
        self._model = model
        self._data = data
        self._verbose = verbose
        self._classifier = classifier
        # outcomes
        self.classes = []
        self.num_classes = 0
        self.train_X = None
        self.train_y = None
        self.vectorizer = None
        self.classifier = None
        self.pipeline = None

    def train(self):
        """
        Preprocesses data, fits a model, and finally saves the model to a file.
        """
        self._preprocess()
        self._build_pipeline()
        self._fit()

        # if "cv_results_" in self.classifier:
        # df = pd.DataFrame.from_dict(self.classifier.cv_results_)
        # print(df.sort_values(by=["rank_test_score"]))

    def _preprocess(self):
        """
        Reads lines from the raw dialect data.
        """
        d = defaultdict(list)

        if self._data:
            data = codecs.open(self._data, "r", "UTF-8")
        else:
            logging.debug("--data not found, assuming input from STDIN")
            data = sys.stdin

        # read first line with column identifiers and ignore
        data.readline()

        for line in data:
            # skip empty lines
            line = line.strip()
            if line == "":
                continue

            X, y = line.split(",")
            d[y].append(X)

        logging.debug("Examples per dialect class:")
        for k, v in d.iteritems():
            logging.debug("%s %d" % (k, len(v)))
        logging.debug("Total messages: %d\n" %
                      sum([len(v) for v in d.values()]))

        self.classes = d.keys()
        self.classes.sort()
        self.num_classes = len(self.classes)

        l = []
        logging.debug("Samples from the data:")
        for k, values in d.iteritems():
            logging.debug("%s\t%s" % (values[0], k))
            for value in values:
                l.append((value.lower(), k))

        # shuffle, just to be sure
        random.shuffle(l)
        self.train_X, self.train_y = zip(*l)

    def _build_pipeline(self):
        """
        Builds an sklearn Pipeline. The pipeline consists of a kind of
        vectorizer, followed by a kind of classifier.
        """
        self.vectorizer = CountVectorizer(
            analyzer="char_wb", ngram_range=(2, 6))
        # voc = []
        # with open("featureDB.final.csv") as featureDB:
        #    for wordLine in featureDB:
        #        word = wordLine.split(',')[0]
        #        if word not in voc:
        #            voc.append(word)
#
        # self.vectorizer = CountVectorizer(vocabulary=voc)
        if self._classifier == "mlp":
            param_grid = {'hidden_layer_sizes': [30, 40]}

            # solver: adam, alpha: 0.001, activation: relu, hidden layers: 40
            mlp = MLPClassifier(early_stopping=True,
                                alpha=0.0001, hidden_layer_sizes=40, verbose=True)

            self.classifier = mlp

            # self.classifier = GridSearchCV(
            #    mlp, param_grid, cv = 5, scoring = 'accuracy', n_jobs = -1, return_train_score = True, verbose = 10)
            # self.classifier = RandomizedSearchCV(
            #    mlp, param_grid, cv=10, scoring='accuracy', n_jobs=1, return_train_score=True)
        elif self._classifier == "svm":
            #{u'kernel': u'rbf', u'C': 10, u'gamma': 0.001}
            #{u'kernel': u'rbf', u'C': 9, u'gamma': 0.0009}
            # C=5, gamma=0.0005
            param_grid = {'C': np.arange(4, 6, 1), 'gamma': np.arange(
                0.0004, 0.0006, 0.0001), 'kernel': ['rbf']},

            svm = SVC(C=5, gamma=0.0005)
            self.classifier = svm
            # self.classifier = GridSearchCV(
            #    svm, param_grid, cv=10, scoring='accuracy', n_jobs=-1, return_train_score=True, verbose=10)

        elif self._classifier == "gradient":
            param_grid = {'learning_rate': [
                0.8], 'n_estimators': [500], 'max_depth': [4]},

            gradient = GradientBoostingClassifier(
                learning_rate=0.4, max_depth=4, n_estimators=500)

            self.classifier = gradient
            # self.classifier = GridSearchCV(
            #    gradient, param_grid, cv=5, scoring='accuracy', n_jobs=-1, return_train_score=True, verbose=10)
        elif self._classifier == "random_forest":
            param_grid = {
                'n_estimators': [50, 80, 100],
                'max_features': ["sqrt", "auto", "log2"]
            }
            rfc = RandomForestClassifier(
                verbose=True, n_estimators=80, n_jobs=-1)

            self.classifier = GridSearchCV(
                rfc, param_grid, cv=10, scoring='accuracy', n_jobs=-1, return_train_score=True, verbose=10)
        else:
            self.classifier = DummyClassifier(strategy="stratified")

        self.pipeline = Pipeline([
            ("vectorizer", self.vectorizer),
            # Boost n-grams
            ("select", SelectKBest(chi2, k=30000)),
            ("cleaning", CleanTransformer()),
            ("tfidf", TfidfTransformer()),
            ("clf", self.classifier)
        ])

        logging.debug(self.vectorizer)
        logging.debug(self.classifier)
        logging.debug(self.pipeline)

    def _fit(self):
        """
        Fits a model for the preprocessed data.
        """
        self.pipeline.fit(self.train_X, self.train_y)

    def save(self):
        """
        Save the whole pipeline to a pickled file.
        """
        from sklearn.externals import joblib
        joblib.dump(self.pipeline, self._model)
        logging.debug("Classifier saved to '%s'" % self._model)


class Predictor(object):
    """
    Predicts the dialect of text, given a trained model.
    """

    def __init__(self):
        """
        """

    def _load(self, model):
        """
        Loads a model that was previously trained and saved.
        """
        from sklearn.externals import joblib
        self.pipeline = joblib.load(model)
        logging.debug("Loading model pipeline from '%s'" % model)

    def _predict(self, model, samples, label_only=False):
        """
        Predicts the class (=dialect) of new text samples.
        """
        self._load(model)

        predictions = []

        for sample in samples:
            sample = sample.strip().split(
                ",")[1].lower()  # column 0 is the index
            prediction = self.pipeline.predict([sample])[0]
            """
            if re.search("ang\s", sample) and prediction == "LU":
                prediction = "BE"
            if re.search("[aeiouäöüóòáàéèíìúù]{3,}", sample):
                prediction = "BE"
            # if len(re.findall("[äöüaeiou]\1", sample)) > 2 and (prediction == "BE" or prediction == "BS"):
            #    prediction = "LU"
            if re.search("(imene|en\s)", sample):
                prediction = "LU"
            if re.search("\s.ai\s", sample):
                prediction = "BS"
            """
            if label_only:
                predictions.append(prediction)
            else:
                predictions.append((sample, prediction))

        return predictions

    def predict(self, model, samples, label_only=False, combined=False):
        if not combined:
            return self._predict(model, samples, label_only)

        predictionsMLP = self._predict(
            "model_mlp.pkz", samples, True)
        predictionsSVM = self._predict(
            "model_svm.pkz", samples, True)
        predictionsGRAD = self._predict(
            "model_gradient.pkz", samples, True)

        predictions = np.array(
            [predictionsMLP, predictionsSVM, predictionsGRAD])
        final_predictions = []

        for idx in range(predictions.shape[1]):
            all_predictions = predictions[:, idx]

            found_classes, counts = np.unique(
                all_predictions, return_counts=True)

            frequencies = dict(zip(counts, found_classes))
            if found_classes.shape[0] == 3:
                # If 3 different classes occurred, take the estimation of
                # MLP
                final_predictions.append(found_classes[0])
            elif 2 in frequencies:  # Take the class with the highest frequency
                final_predictions.append(frequencies[2])
            else:
                final_predictions.append(frequencies[3])

        return final_predictions

    def evaluate(self, model, samples, combined):
        """
        Evaluates the classifier with gold labelled data.
        """
        test_y = []
        test_X = []
        for sample in samples:
            sample = sample.strip()
            X, y = sample.split("\t")
            test_y.append(y)
            test_X.append(X)

        logging.debug("Number of gold samples found: %d" % len(test_y))

        predictions = self.predict(
            model, test_X, label_only=True, combined=combined)
        logging.info(metrics.classification_report(test_y, predictions,
                                                   target_names=None))

        logging.info("Accuracy: " +
                     str(metrics.accuracy_score(test_y, predictions)))


def parse_cmd():
    parser = argparse.ArgumentParser(
        description="train a classifier for dialect data and use it for predictions")

    parser.add_argument(
        "-m", "--model",
        type=str,
        required=False,
        help="if --train, then save model to this path. If --predict, use saved model at this path."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        required=False,
        help="write verbose output to STDERR (default: False)"
    )

    mode_options = parser.add_mutually_exclusive_group(required=True)
    mode_options.add_argument(
        "--train",
        action="store_true",
        required=False,
        help="train a new model and save to the path -m/--model"
    )
    mode_options.add_argument(
        "--predict",
        action="store_true",
        required=False,
        help="predict classes of new samples, write predicted classes to STDOUT"
    )
    mode_options.add_argument(
        "--evaluate",
        action="store_true",
        required=False,
        help="evaluate trained model, write report to STDOUT. If --evaluate, data in --samples is assumed to include the gold label"
    )

    train_options = parser.add_argument_group("training parameters")

    train_options.add_argument(
        "--data",
        type=str,
        required=False,
        help="path to file with raw dialect data, UTF-8. If --data is not given, input from STDIN is assumed"
    )
    train_options.add_argument(
        "--classifier",
        type=str,
        required=False,
        default="mlp",
        help="type of classifier to be trained. Either 'mlp' or 'dummy' (stratified class probabilities)",
        choices=("mlp", "svm", "gradient", "crf", "random_forest", "dummy")
    )

    predict_options = parser.add_argument_group("prediction parameters")

    predict_options.add_argument(
        "--samples",
        type=str,
        required=False,
        help="Path to file containing samples for which a class should be predicted. If --samples is not given, input from STDIN is assumed"
    )

    predict_options.add_argument(
        "--combined",
        required=False,
        action="store_true",
        help="Whether to combine results of the different classifiers. The --classifier parameter will be ignored."
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_cmd()

    # set up logging
    if args.verbose:
        level = logging.DEBUG
    elif args.evaluate:
        level = logging.INFO
    else:
        level = logging.WARNING
    logging.basicConfig(level=level, format='%(levelname)s: %(message)s')

    if args.train:
        t = Trainer(model=args.model,
                    data=args.data,
                    verbose=args.verbose,
                    classifier=args.classifier
                    )
        t.train()
        t.save()
    else:
        p = Predictor()
        if args.samples:
            input_ = codecs.open(args.samples, "r", "UTF-8")
        else:
            logging.debug("--samples not found, assuming input from STDIN")
            input_ = sys.stdin

        # read first line and ignore, column names
        input_.readline()

        if args.evaluate:
            p.evaluate(model=args.model, samples=input_,
                       combined=args.combined)
        else:
            predictions = p.predict(
                model=args.model, samples=input_, label_only=True, combined=args.combined)
            print "Id,Prediction"
            for index, prediction in enumerate(predictions):
                print "%s,%s" % (index + 1, prediction)


if __name__ == '__main__':
    main()
