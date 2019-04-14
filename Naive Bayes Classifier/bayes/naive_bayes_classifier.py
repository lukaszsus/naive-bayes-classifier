import numpy as np

from bayes.probability_holder import ProbabilityHolder


class NaiveBayesClassifier:
    def __init__(self, X=None, Y=None):
        if X is not None and Y is not None:
            self.prob_holder = ProbabilityHolder(X, Y)
            self._avail_y = range(min(Y), max(Y) + 1)

    def fit(self, X, Y):
        self.prob_holder = ProbabilityHolder(X, Y)
        self._avail_y = range(min(Y), max(Y) + 1)

    def predict(self, X: np.ndarray):
        predicted = list()
        for x in X:
            predicted.append(self._predict_one_example(x))
        return np.asarray(predicted)

    def _predict_one_example(self, x: np.ndarray):
        certainity_for_ys = list()
        for y in self._avail_y:     # for every clas
            certainity_for_ys.append(self.prob_holder.p_y[y])
            for i in range(len(x)):      # for every feature
                certainity_for_ys[-1] *= self.prob_holder.get(i, x[i], y)
        return self._avail_y[certainity_for_ys.index(max(certainity_for_ys))]

    def print_prob(self):
        print(self.prob_holder.p_y)
        for matrix in self.prob_holder.p_x_y:
            print(matrix)

