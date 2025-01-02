# -*- coding: utf-8 -*-
"""
@author: Klaus Hajdaraj
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import yaml
import joblib
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from typing import Optional, Tuple, Union

# Read file paths
with open("config.yaml", "r") as file:
    file_path = yaml.safe_load(file)

# Read the path to my Documents folder in my computer
documents_file_path = file_path["paths"]["documents"]

# Set path to results for full set of treatments (all six), or the subset of treatments (1, 2, 4, 5)
results_full_set_path = documents_file_path + "repetitions_alltreatments.joblib"
results_sub_set_path = documents_file_path + "repetitions_subsettreatments.joblib"

# Set the theme for all graphs
sns.set_theme(style="white")


def get_data(full_data: bool = True) -> pd.DataFrame:
    if full_data:
        filename = results_full_set_path
    else:
        filename = results_sub_set_path
    with open(filename, "rb") as file:
        CV_Results = joblib.load(file)
    estimators = [x for x in CV_Results.overall_stats.index if x.endswith("Y")]
    df_y = pd.DataFrame(columns=estimators)
    for estimator in df_y:
        results = []
        for repetition in CV_Results.rep_stats:
            results.append(repetition.loc[estimator, "overall"])
        df_y[estimator] = results
    global MEAN
    MEAN = CV_Results.overall_stats.loc["AllY", "overall"]
    return df_y


def makeplot(data: pd.DataFrame, ax: Optional[Axes] = None) -> None:
    sns.set_theme(style="white")
    percent_mean, percent_t4 = get_percentages(data)
    if ax:
        plot = sns.histplot(data, kde=True, bins=10, ax=ax, color="lightslategrey")
        ax.axvline(MEAN, 0, 0.95, color="red", linestyle="--", linewidth=3)
        ax.axvline(1970, 0, 0.95, color="green", linestyle="--", linewidth=3)
        ax.set_title(f"{percent_mean} over mean, {percent_t4} over treatment 4")
    else:
        plot = sns.histplot(data, bins=10, kde=True, color="lightslategrey")
        plt.axvline(MEAN, 0, 0.95, color="red", linestyle="--", linewidth=3)
        plt.axvline(1970, 0, 0.95, color="green", linestyle="--", linewidth=3)
        plot.fig.suptitle(f"{percent_mean} over mean, {percent_t4} over treatment 4")


def get_percentages(data: Union[pd.Series, np.ndarray]) -> Tuple[str, str]:
    n = len(data)
    percent_mean = str(sum(data >= MEAN)) + "/" + str(n)
    percent_t4 = str(sum(data > 1969.846698)) + "/" + str(n)
    return (percent_mean, percent_t4)


if __name__ == "__main__":
    df_y = get_data()
    for column in df_y:
        print(column)
        makeplot(df_y[column])
        get_percentages(df_y[column])
