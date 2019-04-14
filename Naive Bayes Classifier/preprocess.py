import numpy as np
from sklearn.cluster import KMeans


def remove_equal_or_decreasing_successor(bins: np.ndarray):
    to_remove = list()
    for i in range(len(bins)-1):
        if bins[i] >= bins[i+1]:
            to_remove.append(i+1)

    bins = np.delete(bins, to_remove)
    return bins


def discretize_equal_width(X, num_bins):
    columns = X.transpose()
    for col in columns:
        # if type(col[0]) == float or type(col[0]) == np.float64:
        interval = float((max(col) - min(col)))/num_bins
        bins = np.asarray([min(col) + i*interval for i in range(num_bins+1)])
        bins = remove_equal_or_decreasing_successor(bins)
        bins[-1] = bins[-1] + 0.01
        col[:] = np.digitize(col, bins)
    return np.int_(columns.transpose())


def discretize_equal_freq(X, num_bins):
    columns = X.transpose()
    interval = int(1.0/num_bins*100.0)
    for col in columns:
        # if type(col[0]) == float or type(col[0]) == np.float64:
        bins = [np.percentile(col, interval*i) for i in range(num_bins+1)]
        bins = remove_equal_or_decreasing_successor(bins)
        bins[-1] = bins[-1] + 0.01
        col[:] = np.digitize(col, bins)
    return np.int_(columns.transpose())


def discretize_k_means(X, num_bins):
    kmeans = KMeans(n_clusters=num_bins, random_state=0)        # clustering with Lloyd algorithm
    columns = X.transpose()
    for col in columns:
        # if type(col[0]) == float or type(col[0]) == np.float64:
        x = col.copy()
        x[:] = kmeans.fit_predict(x.reshape(-1, 1))
        col[:] = x
    return np.int_(columns.transpose())


def make_start_from_zero_integers(vector):
    vector = np.int_(vector)
    vector = vector - min(vector)
    return vector


def translate_class_labels(Y: np.ndarray):
    avail_y = list()
    for y in Y:
        if not y in avail_y:
            avail_y.append(y)
    avail_y.sort()
    return np.int_([avail_y.index(y) for y in Y])
