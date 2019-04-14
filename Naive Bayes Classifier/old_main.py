import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn import datasets, metrics

from bayes.naive_bayes_classifier import NaiveBayesClassifier
from bayes.naive_bayes_gaussian_classifier import NaiveBayesGaussianClassifier
from preprocess import *
from dsloader import *


if __name__ == "__main__":
    skf = StratifiedKFold(n_splits=5, shuffle=True)
    # data, target = load_iris()
    # data, target = load_diabetes()
    # data, target = load_glass()
    data, target = load_wine()

    # dicretization of continuous function
    num_bins = 10
    # data = discretize_equal_width(data, num_bins)
    # data = discretize_equal_freq(data, num_bins)
    # data = discretize_k_means(data, num_bins)
    target = translate_class_labels(target)

    # splitting using stratified k-fold cross validation
    split_set_generator = skf.split(data, target)

    # trainning and testing
    y_pred = list()
    y_true = list()

    for train_indices, test_indices in split_set_generator:
        X_train = data[train_indices]
        Y_train = target[train_indices]
        # bayes = NaiveBayesClassifier(X_train, Y_train)
        bayes = NaiveBayesGaussianClassifier(X_train, Y_train)
        y_pred.extend(bayes.predict(data[test_indices]))
        y_true.extend(target[test_indices])

    print("Confusion matrix:", metrics.confusion_matrix(y_true, y_pred))
    print("Accuracy:", metrics.accuracy_score(y_true, y_pred))
    print("Precision:", metrics.precision_score(y_true, y_pred, average=None))
    print("Recall:", metrics.recall_score(y_true, y_pred, average=None))
    print("F1 score:", metrics.f1_score(y_true, y_pred, average=None))

    # bayes = NaiveBayesClassifier()
    # cross_validate(bayes, data, target, cv=10, scoring=['accuracy', 'precision', 'recall', 'f1_samples'])
