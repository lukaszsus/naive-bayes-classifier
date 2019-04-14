import numpy as np

from bayes.probability_holder import ProbabilityHolder
from bayes.gauss_params_holder import GaussParamsHolder


class NaiveBayesGaussianClassifier:
    def __init__(self, X=None, Y=None):
        self.prob_holder: ProbabilityHolder = None
        self.gauss_holder: GaussParamsHolder = None
        self._avail_y: np.ndarray = None
        self._use_gauss: list = None
        if X is not None and Y is not None:
            self.fit(X, Y)

    def fit(self, X, Y):
        self._X = X
        self._Y = Y
        self._avail_y = range(min(Y), max(Y) + 1)
        self._use_gauss: list = None
        self._check_when_use_gauss()
        self.prob_holder = ProbabilityHolder(X, Y, [not i for i in self._use_gauss])
        self.gauss_holder = GaussParamsHolder(X, Y, self._use_gauss)

    def _check_when_use_gauss(self):
        self._use_gauss = list()
        for i in range(self._X.shape[1]):
            if type(self._X[0, i]) == np.float64:
                self._use_gauss.append(True)
            else:
                self._use_gauss.append(False)

    def predict(self, X: np.ndarray):
        predicted = list()
        for i in range(len(X)):
            predicted.append(self._predict_one_example(i, X[i, :]))
        return np.asarray(predicted)

    def _predict_one_example(self, i: int, x: np.ndarray):
        certainity_for_ys = list()
        for y in self._avail_y:  # for every class
            certainity_for_ys.append(self.prob_holder.p_y[y])
            for i in range(len(x)):  # for every feature
                if self._use_gauss[i]:
                    certainity_for_ys[-1] *= self.gauss_holder.get(feature_index=i, on_condition_y=y, value=x[i])
                else:
                    certainity_for_ys[-1] *= self.prob_holder.get(i, x[i], y)
        return self._avail_y[certainity_for_ys.index(max(certainity_for_ys))]
