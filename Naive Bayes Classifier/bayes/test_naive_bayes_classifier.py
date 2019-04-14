import numpy as np
from unittest import TestCase

from bayes.naive_bayes_classifier import NaiveBayesClassifier


class TestNaiveBayesClassifier(TestCase):
    def test_train(self):
        bayes = NaiveBayesClassifier()
        X = np.asarray([[1, 1, 1],
             [2, 2, 2],
             [3, 3, 3]])
        Y = np.asarray([1, 2, 3])
        bayes.train(X, Y)
        self.fail()
