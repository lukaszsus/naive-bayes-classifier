import os
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

OUTCOME_PATH = '/home/lukasz/Developer/VIII_semestr/IMAD/L1/outcomes/'


def plot_num_bins_comparision(data: pd.DataFrame):
    datasets = data["dataset"].unique()
    disc_methods = data["disc_method"].unique()
    num_bins = data["num_bins"].unique()
    metrics_names = ['acc_mean', 'prec_mean', 'rec_mean', 'f1_mean']

    for ds in datasets:
        ds_data = data[data['dataset'] == ds]
        # for disc_method in disc_methods:
        #     ds_disc = ds_data[ds_data['disc_method'] == disc_method]
        for metric_name in metrics_names:
            gauss_val = get_gauss_val(ds, metric_name)

            name = ds + "_" + metric_name + "_discretization_methods"
            title = ds + " " + metric_name + " as a function of number of bins"
            fig, ax = plt.subplots()
            ax = sns.scatterplot(x="num_bins", y=metric_name, hue="disc_method", data = ds_data)
            ax.set_title(title)
            ax.plot([0, 10], [gauss_val, gauss_val], 'b--', linewidth=1)
            path = os.path.join(OUTCOME_PATH, "plots")
            path_pdf = os.path.join(path, name + ".pdf")
            path_png = os.path.join(path, name + ".png")
            fig.savefig(path_pdf, bbox_inches='tight')
            fig.savefig(path_png, bbox_inches='tight')


def get_gauss_val(ds, metric_name):
    file_name = "gauss_sample.csv"
    path = os.path.join(OUTCOME_PATH, file_name)
    data = pd.read_csv(path)
    data = data[data["dataset"] == ds]
    return data[metric_name].values[0]


if __name__ == '__main__':
    file_name = "data_without_gauss.csv"
    path = os.path.join(OUTCOME_PATH, file_name)
    data = pd.read_csv(path)

    plot_num_bins_comparision(data)
