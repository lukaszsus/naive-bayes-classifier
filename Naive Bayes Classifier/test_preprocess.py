from unittest import TestCase

import numpy as np

from preprocess import discretize_equal_width, discretize_equal_freq


class TestPreprocess(TestCase):
    def test_discretize_equal_width(self):
        X = np.asarray([[0.0, 0.1, 0.7, 0.75, 2.3, 2.9, 3.0],
                        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]])
        X = X.transpose()
        disc = discretize_equal_width(X, 3)
        print(disc)
        self.fail()

    def test_discretize_equal_freq(self):
        X = np.asarray([[0.0, 0.1, 0.7, 0.75, 2.3, 2.9, 3.0],
                        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6]])
        X = X.transpose()
        disc = discretize_equal_freq(X, 3)
        print(disc)
        self.fail()
