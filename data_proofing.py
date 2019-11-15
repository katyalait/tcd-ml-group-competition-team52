import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pre_process as pp

"""
For analysing the data and getting visual insights into the values 

"""

def correlation_matrix(df):
    """
    Calculates and displays the correlation matrix of the dataframe
    :param df: dataframe
    :return:
    """
    corr = df.corr()
    ax = sns.heatmap(
        corr,
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    );
    plt.show()

def get_outliers(df, col):
    """
    Gets the z-scores of the values of the provided col name and outputs the
    size of the dataframe after these rows have been removed
    :param df: dataframe
    :param col: column name
    :return:
    """
    ax = df[col].plot.hist(bins=40, alpha=0.5)
    plt.show()

    z = np.abs(stats.zscore(df[col]))
    df_w_o_outliers = df[(z < 6)]


    ax = df_w_o_outliers[col].plot.hist(bins=40, alpha=0.5)
    plt.show()
    return df_w_o_outliers

def category_histogram(df, col):
    counts = df[col].value_counts()
    ax = counts.plot(kind='bar')
    plt.show()
