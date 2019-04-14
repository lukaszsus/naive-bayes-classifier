import os
import re
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB

from bayes.naive_bayes_classifier import NaiveBayesClassifier
from bayes.naive_bayes_gaussian_classifier import NaiveBayesGaussianClassifier
from plotter import OUTCOME_PATH
from preprocess import *
from dsloader import *

warnings.filterwarnings('ignore')


class Researcher:
    NUM_TESTS_PER_EXAMPLE = 100
    NUM_FOLDS = (2, 3, 5, 10)
    NUM_BINS = (2, 3, 5, 10)
    DATASET_INDICES = {"diabetes": 0, "glass": 1, "wine": 2}
    # COLUMN_LIST = ["dataset", "fold-class", "n-folds", "acc", "prec", "rec", "f1"]
    METRICS_COLUMN_LIST = ["dataset", "fold-class", "n-folds", "num_bins", "acc_mean", "acc_var", "prec_mean", "prec_var",
                   "rec_mean", "rec_var", "f1_mean", "f1_var"]
    METRICS_COLUMN_LIST_MEANS = ["dataset", "fold-class", "n-folds", "disc_method", "num_bins", "acc_mean", "prec_mean",
                   "rec_mean", "f1_mean"]
    DISC_METHODS = (discretize_equal_width, discretize_equal_freq, discretize_k_means)

    def __init__(self):
        self._accuracies = None
        self._precisions = None
        self._recalls = None
        self._f1_scores = None

        self._mean_acc = None
        self._var_acc = None
        self._mean_prec = None
        self._var_prec = None
        self._mean_rec = None
        self._var_rec = None
        self._mean_f1 = None
        self._var_f1 = None

        self._splitter = None
        self._n_folds = None
        self._loader = None
        self._num_bins_multiplier = None
        self._disc_method = None
        self.use_gauss = True
        self.use_lib_bayes = False

        # self._outcomes: pd.DataFrame = pd.DataFrame(columns=self.COLUMN_LIST)
        self._metrics: pd.DataFrame = pd.DataFrame(columns=self.METRICS_COLUMN_LIST)

    def load_datasets(self):
        datasets_loader = list()
        datasets_loader.append(load_diabetes)
        datasets_loader.append(load_glass)
        datasets_loader.append(load_wine)
        return datasets_loader

    def do_crossval_research(self, data: np.ndarray, target: np.ndarray, stratified: bool):
        if stratified:
            self._splitter = StratifiedKFold(n_splits=self._n_folds, shuffle=True)
        else:
            self._splitter = KFold(n_splits=self._n_folds, shuffle=True)
        split_set_generator = self._splitter.split(data, target)

        # trainning and testing
        y_pred = list()
        y_true = list()

        for train_indices, test_indices in split_set_generator:
            X_train = data[train_indices]
            Y_train = target[train_indices]
            if self.use_lib_bayes:
                bayes = MultinomialNB()
                bayes.fit(X_train, Y_train)
            elif self.use_gauss:
                bayes = NaiveBayesGaussianClassifier(X_train, Y_train)
            else:
                bayes = NaiveBayesClassifier(X_train, Y_train)
            y_pred.extend(bayes.predict(data[test_indices]))
            y_true.extend(target[test_indices])

        confusion = metrics.confusion_matrix(y_true, y_pred)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_pred, average=None)
        recall = metrics.recall_score(y_true, y_pred, average=None)
        f1_score = metrics.f1_score(y_true, y_pred, average=None)

        return {"confusion": confusion, "accuracy": accuracy, "precision": precision,
                "recall": recall, "f1_score": f1_score}

    def make_n_samples(self, data, target, stratified: bool = True):
        self._accuracies = list()
        self._precisions = list()
        self._recalls = list()
        self._f1_scores = list()

        for i in range(self.NUM_TESTS_PER_EXAMPLE):
            metrics = self.do_crossval_research(data, target, stratified)

            self._accuracies.append(metrics["accuracy"])
            self._precisions.append(metrics["precision"])
            self._recalls.append(metrics["recall"])
            self._f1_scores.append(metrics["f1_score"])

        self._accuracies = np.asarray(self._accuracies)
        self._precisions = np.asarray(self._precisions)
        self._recalls = np.asarray(self._recalls)
        self._f1_scores = np.asarray(self._f1_scores)

        # record = pd.DataFrame([[self.__get_name_from_loader(),
        #                         type(self._splitter).__name__,
        #                         self._n_folds,
        #                         self._accuracies[i],
        #                         self._precisions[i],
        #                         self._recalls[i],
        #                         self._f1_scores[i]]
        #                         for i in range(len(self._accuracies))], columns=self.COLUMN_LIST)
        # if self._outcomes.empty:
        #     self._outcomes = record
        # else:
        #     self._outcomes = pd.concat([self._outcomes, record], ignore_index=True)

    def make_n_samples_discretized(self, data, target, stratified: bool = True):
        self._accuracies = list()
        self._precisions = list()
        self._recalls = list()
        self._f1_scores = list()

        for i in range(self.NUM_TESTS_PER_EXAMPLE):
            metrics = self.do_crossval_research(data, target, stratified)

            self._accuracies.append(metrics["accuracy"])
            self._precisions.append(metrics["precision"])
            self._recalls.append(metrics["recall"])
            self._f1_scores.append(metrics["f1_score"])

        self._accuracies = np.asarray(self._accuracies)
        self._precisions = np.asarray(self._precisions)
        self._recalls = np.asarray(self._recalls)
        self._f1_scores = np.asarray(self._f1_scores)

    def refresh_metrics_summary(self, num_lists=2):
        self._mean_acc = [list() for i in range(num_lists)]
        self._var_acc = [list() for i in range(num_lists)]
        self._mean_prec = [list() for i in range(num_lists)]
        self._var_prec = [list() for i in range(num_lists)]
        self._mean_rec = [list() for i in range(num_lists)]
        self._var_rec = [list() for i in range(num_lists)]
        self._mean_f1 = [list() for i in range(num_lists)]
        self._var_f1 = [list() for i in range(num_lists)]

    def metrics_summary_folds(self):
        mean_acc = np.mean(self._accuracies)
        mean_prec = np.mean(self._precisions)
        mean_rec = np.mean(self._recalls)
        mean_f1 = np.mean(self._f1_scores)
        var_acc = np.mean(np.var(self._accuracies, 0))
        var_prec = np.mean(np.var(self._precisions, 0))
        var_rec = np.mean(np.var(self._recalls, 0))
        var_f1 = np.mean(np.var(self._f1_scores, 0))

        record = pd.DataFrame([[self._loader.__name__,
                               type(self._splitter).__name__,
                               self._n_folds,
                               self._num_bins_multiplier,
                               mean_acc, var_acc,
                               mean_prec, var_prec,
                               mean_rec, var_rec,
                               mean_f1, var_f1]], columns=self.METRICS_COLUMN_LIST)
        if self._metrics.empty:
            self._metrics = record
        else:
            self._metrics = pd.concat([self._metrics, record], ignore_index=True)

        index = 0 if type(self._splitter) == StratifiedKFold else 1
        self._mean_acc[index].append(mean_acc)
        self._var_acc[index].append(var_acc)
        self._mean_prec[index].append(mean_prec)
        self._var_prec[index].append(var_prec)
        self._mean_rec[index].append(mean_rec)
        self._var_rec[index].append(var_rec)
        self._mean_f1[index].append(mean_f1)
        self._var_f1[index].append(var_f1)

        print("Dataset: {0}\nFold-class: {1}".format(self.__get_name_from_loader(), type(self._splitter).__name__))
        print("N-folds: {0}\n".format(self._n_folds))
        print("Accuracy:\nmean: {0}\tvariance: {1}".format(mean_acc, var_acc))
        print("Precision:\nmean: {0}\tvariance: {1}".format(mean_prec, var_prec))
        print("Recall:\nmean: {0}\tvariance: {1}".format(mean_rec, var_rec))
        print("F1 score:\nmean: {0}\tvariance: {1}".format(mean_f1, var_f1))
        print()

    def metrics_summary(self):
        mean_acc = np.mean(self._accuracies)
        mean_prec = np.mean(self._precisions)
        mean_rec = np.mean(self._recalls)
        mean_f1 = np.mean(self._f1_scores)

        record = pd.DataFrame([[self._loader.__name__,
                               type(self._splitter).__name__,
                               self._n_folds,
                               self._get_disc_method_name(),
                               self._num_bins_multiplier,
                               mean_acc,
                               mean_prec,
                               mean_rec,
                               mean_f1]], columns=self.METRICS_COLUMN_LIST_MEANS)
        if self._metrics.empty:
            self._metrics = record
        else:
            self._metrics = pd.concat([self._metrics, record], ignore_index=True)

        # index = self.DATASET_INDICES[self.__get_name_from_loader()]
        # self._mean_acc[index].append(mean_acc)
        # self._mean_prec[index].append(mean_prec)
        # self._mean_rec[index].append(mean_rec)
        # self._mean_f1[index].append(mean_f1)

        print("Dataset: {0}\nFold-class: {1}".format(self.__get_name_from_loader(), type(self._splitter).__name__))
        print("N-folds: {0}\nDisc. method: {1}".format(self._n_folds, self._get_disc_method_name()))
        print("N-bins: {0}".format(self._num_bins_multiplier))
        print("Accuracy:\nmean: {0}".format(mean_acc))
        print("Precision:\nmean: {0}".format(mean_prec))
        print("Recall:\nmean: {0}".format(mean_rec))
        print("F1 score:\nmean: {0}".format(mean_f1))
        print()

    def _save_to_file(self):
        path = os.path.join(OUTCOME_PATH, self.__get_name_from_loader() + ".csv")
        self._metrics.to_csv(path, index=False)
        #
        # path = path = os.path.join(OUTCOME_PATH, self.__get_name_from_loader() + "_all.csv")
        # self._outcomes.to_csv(path, index=False)

    def __get_name_from_loader(self):
        name = self._loader.__name__
        name = re.search("_.*", name)
        name = name.group(0)[1:]
        return name

    def _plot_metrics_over_folds(self):
        dataset_name = self.__get_name_from_loader()
        self._save_plots_over_folds(dataset_name, self._mean_acc, "mean_acc")
        self._save_plots_over_folds(dataset_name, self._var_acc, "var_acc", 'mean variance')

        self._save_plots_over_folds(dataset_name, self._mean_prec, "mean_prec")
        self._save_plots_over_folds(dataset_name, self._var_prec, "var_prec", 'mean variance')

        self._save_plots_over_folds(dataset_name, self._mean_rec, "mean_rec")
        self._save_plots_over_folds(dataset_name, self._var_rec, "var_rec", 'mean variance')

        self._save_plots_over_folds(dataset_name, self._mean_f1, "mean_f1")
        self._save_plots_over_folds(dataset_name, self._var_f1, "var_f1", 'mean variance')

    def _save_plots_over_folds(self, dataset_name, data, metric_name, y_label = 'mean'):
        fig, ax = plt.subplots()
        ax.plot(self.NUM_FOLDS, data[0], 'r.', label='StratifiedKFold')
        ax.plot(self.NUM_FOLDS, data[1], 'bs', label='KFold')
        ax.set_title(metric_name)
        ax.set_xlabel('number of folds')
        ax.set_ylabel(y_label)
        path = OUTCOME_PATH
        path = os.path.join(path, "plots")
        path_pdf = os.path.join(path, dataset_name + " " + metric_name + ".pdf")
        path_png = os.path.join(path, dataset_name + " " + metric_name + ".png")

        legend = ax.legend(shadow=True)

        fig.savefig(path_pdf, bbox_inches='tight')
        fig.savefig(path_png, bbox_inches='tight')

    def go_over_folds(self):
        """Research on number of folds is conducted on Gaussian Naive Bayes Classifier."""
        self.use_gauss = True
        self.use_lib_bayes = False

        dataset_loaders = self.load_datasets()
        for self._loader in dataset_loaders:
            self.refresh_metrics_summary()
            for self._n_folds in self.NUM_FOLDS:
                data, target = self._loader()
                target = translate_class_labels(target)
                self.make_n_samples(data, target, stratified=False)
                self.metrics_summary_folds()

                self.make_n_samples(data, target, stratified=True)
                self.metrics_summary_folds()
            self._save_to_file()
            self._plot_metrics_over_folds()
            # plot_boxplots_over_folds(self._outcomes)

    def go_over_discretizations(self):
        self.use_gauss = False

        self._n_folds = 5
        dataset_loaders = self.load_datasets()
        for self._loader in dataset_loaders:
            self.refresh_metrics_summary()

            self._splitter = StratifiedKFold(n_splits=5, shuffle=True)
            data, target = self._loader()
            target = translate_class_labels(target)
            # num_classes = len(np.unique(target))
            for self._num_bins_multiplier in self.NUM_BINS:
                num_bins = self._num_bins_multiplier * 1 # num_classes
                for self._disc_method in self.DISC_METHODS:
                    data_discretized = self._disc_method(data.copy(), num_bins)
                    self.make_n_samples_discretized(data_discretized, target, self._n_folds)
                    self.metrics_summary()
            self._save_to_file()

    def go_over_gauss_sample(self):
        self.use_gauss = True

        self._n_folds = 5
        dataset_loaders = self.load_datasets()
        for self._loader in dataset_loaders:
            self.refresh_metrics_summary()

            self._splitter = StratifiedKFold(n_splits=5, shuffle=True)
            data, target = self._loader()
            target = translate_class_labels(target)
            self.make_n_samples(data, target)
            self.metrics_summary()
            self._save_to_file()

    def go_over_discretization_with_lib_bayes(self):
        self.use_gauss = False
        self.use_lib_bayes = True

        self._n_folds = 5
        dataset_loaders = self.load_datasets()
        for self._loader in dataset_loaders:
            self.refresh_metrics_summary()

            self._splitter = StratifiedKFold(n_splits=5, shuffle=True)
            data, target = self._loader()
            target = translate_class_labels(target)
            for self._num_bins_multiplier in self.NUM_BINS:
                num_bins = self._num_bins_multiplier * 1  # num_classes
                for self._disc_method in self.DISC_METHODS:
                    data_discretized = self._disc_method(np.copy(data), num_bins)
                    self.make_n_samples_discretized(data_discretized, target)
                    self.metrics_summary()
            self._save_to_file()

    def _get_disc_method_name(self):
        if self.use_gauss:
            return "gauss"
        return self._disc_method.__name__
