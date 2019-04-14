import numpy as np


class GaussParamsHolder:
    def __init__(self, X, Y, use: list = None):
        self._X: np.ndarray = X
        self._Y: np.ndarray = Y
        self._use = use
        self.means: list = None   # feature x class
        self.variances: list = None  # feature x class
        self._avail_y: list = None    # list of available classes (as numbers)
        self._total_set_quantity: int = None

        if use is None:
            self._use = [True for i in range(len(self._Y))]
        self._count_gauss_params()

    def _count_gauss_params(self):
        self.means: list = list()
        self.variances: list = list()
        self._avail_y = range(min(self._Y), max(self._Y) + 1)
        for feature_index in range(self._X.shape[1]):
            if self._use[feature_index]:
                feature_variance = list()
                feature_mean = list()
                for y_index in self._avail_y:
                    x = list()
                    for k in range(self._X.shape[0]):
                        if self._Y[k] == y_index:
                            x.append(self._X[k,feature_index])
                    x = np.asarray(x)
                    feature_mean.append(np.mean(x))
                    var_x = np.var(x)
                    if var_x == 0:      # smoothing
                        var_x = 0.001
                    feature_variance.append(var_x)
                self.means.append(np.asarray(feature_mean))
                self.variances.append(np.asarray(feature_variance))
            else:
                self.means.append(np.asarray([float('nan') for i in range(len(self._avail_y))]))
                self.variances.append(np.asarray([float('nan') for i in range(len(self._avail_y))]))
        self.means = np.asarray(self.means)
        self.variances = np.asarray(self.variances)

    # def _count_every_feature_cond_prob(self):
    #     p_x_y = list()
    #     for feature_index in range(self._X.shape[1]):
    #         p_x_y.append(self._count_feature_cond_prob(feature_index))
    #     self.p_x_y = np.asarray(p_x_y)
    #
    # def _count_feature_cond_prob(self, feature_index: int):
    #     """Count conditional probability:
    #     P(x_i | y)
    #     for feature x_i.
    #     :param feature_index index of feature"""
    #
    #     probs = list()
    #     avail_x = range(min(self._X[:, feature_index]), max(self._X[:, feature_index]) + 1)
    #     for i in avail_x:
    #         probs.append(list())
    #         for class_index in self._avail_y:
    #             x_equal = self._X[:, feature_index] == i
    #             y_equal = self._Y[:] == class_index
    #             num_smpl_xi_and_y = np.count_nonzero(np.logical_and(x_equal,y_equal)) + 1   # smoothing
    #             num_y = np.count_nonzero(self._Y == class_index) + len(avail_x)*len(self._avail_y)      # smoothing
    #             probs[-1].append(num_smpl_xi_and_y / num_y)
    #     return np.asarray(probs)

    def get(self, feature_index, value, on_condition_y):
        return self._prob_using_gauss(feature_index, on_condition_y, value)

    def _prob_using_gauss(self, feature_index, y_index, x_i):
        multiplier = 1 / np.sqrt(2 * np.pi * self.variances[feature_index, y_index])
        exp = - (x_i - self.means[feature_index, y_index]) ** 2 / (2 * self.variances[feature_index, y_index])
        return multiplier * np.power(np.e, exp)
