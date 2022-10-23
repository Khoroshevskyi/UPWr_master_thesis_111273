# This file contain scripts for dimensionality reduction

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap.umap_ as umap


def run_pca(df: pd.DataFrame, confidence: float = 0.90) -> pd.DataFrame:
    """
    run Principal component analysis
    computing the principal components and using them to perform a change of basis on the data,
    sometimes using only the first few principal components and ignoring the rest.
    :param confidence: Confidence of pca [default: 0.90]
    :param df: dataset on which PCA should be run
    :return: pca return
    """
    pca = PCA(confidence)
    data_numeric_pca = pca.fit_transform(df)

    x_pca = pd.DataFrame(data_numeric_pca)
    print("After feature selection with PCA, from 60483 features we left %i features" % (x_pca.shape[1]))

    return x_pca


def run_umap(df, neighbours=10, metric='euclidean'):
    """
    Uniform Manifold Approximation and Projection
    :param df: dataset
    :param neighbours: number of neighbours
    :param metric: metric to be used in umap [default: euclidean]
    :return: umap dataset
    """
    reducer = umap.UMAP(n_neighbors=neighbours, metric=metric)

    embedding = reducer.fit_transform(df)
    return embedding


def variance_select(x: pd.DataFrame, threshold: int = 500):
    """
    Feature selection by variance
    :param x: X dataframe
    :param threshold: Variance threshold
    :return: selected x
    """
    selector = VarianceThreshold(threshold)
    selector.fit(x)

    x_n = X[X.columns[selector.get_support()]]
    return x_n


feature_select_methods = {
    "pca": run_pca,
    "umap": run_umap,
    "var": variance_select,
}
