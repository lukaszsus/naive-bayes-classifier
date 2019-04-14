import numpy as np


class ProbabilityHolder:
    def __init__(self, X, Y, use: list = None):
        self._X: np.ndarray = X
        self._Y: np.ndarray = Y
        self._use = use
        self.p_y: np.ndarray = None
        self.p_x_y: list = None   # list of arrays, one array for one feature
        self._avail_y: list = None    # list of available classes (as numbers)
        self._total_set_quantity: int = None

        if self._use is None:
            self._use = [True for i in range(len(self._Y))]

        self._count_y_prob()
        self._count_every_feature_cond_prob()

    def _count_y_prob(self):
        self._total_set_quantity = len(self._Y)
        self._avail_y = range(min(self._Y), max(self._Y) + 1)
        self.p_y = [np.count_nonzero(self._Y == i) / self._total_set_quantity for i in self._avail_y]

    def _count_every_feature_cond_prob(self):
        p_x_y = list()
        for feature_index in range(self._X.shape[1]):
            if self._use[feature_index]:
                p_x_y.append(self._count_feature_cond_prob(feature_index))
            else:
                p_x_y.append(-0.00000001)
        self.p_x_y = np.asarray(p_x_y)

    def _count_feature_cond_prob(self, feature_index: int):
        """Count conditional probability:
        P(x_i | y)
        for feature x_i.
        :param feature_index index of feature"""

        probs = list()
        avail_x = range(min(self._X[:, feature_index]), max(self._X[:, feature_index]) + 1)
        for i in avail_x:
            probs.append(list())
            for class_index in self._avail_y:
                x_equal = self._X[:, feature_index] == i
                y_equal = self._Y[:] == class_index
                num_smpl_xi_and_y = np.count_nonzero(np.logical_and(x_equal,y_equal)) + 1   # smoothing
                num_y = np.count_nonzero(self._Y == class_index) + len(avail_x)*len(self._avail_y)      # smoothing
                probs[-1].append(num_smpl_xi_and_y / num_y)
        return np.asarray(probs)

    def get(self, i, x_i, on_condition_y):
        if 0 <= x_i - 1 < len(self.p_x_y[i]):
            cond_probs = self.p_x_y[i]
            return cond_probs[x_i-1, on_condition_y]
        else:
            return 0
