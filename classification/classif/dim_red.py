# This file contain scripts for dimensionality reduction

from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap.umap_ as umap
import pandas as pd



def no_dim_red(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return dataframe as it is
    :param df: dataframe
    :return: same dataframe
    """
    return df


# def clean_dataset(df):
#     assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
#     df.dropna(inplace=True)
#     indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
#     return df[indices_to_keep].astype(np.float64)

def run_pca(df: pd.DataFrame, confidence: float = 0.90) -> pd.DataFrame:
    """
    run Principal component analysis
    computing the principal components and using them to perform a change of basis on the data,
    sometimes using only the first few principal components and ignoring the rest.
    :param confidence: Confidence of pca [default: 0.90]
    :param df: dataset on which PCA should be run
    :return: pca return
    """
    # # Create the Scaler object
    # scaler = StandardScaler()
    # #
    # data_numeric = pd.DataFrame(df)
    # # Fit your data on the scaler object
    # data_numeric_standardized = scaler.fit_transform(data_numeric)
    # data_numeric_standardized = pd.DataFrame(data_numeric_standardized, columns=data_numeric.columns)
    # data_numeric_standardized = clean_dataset(df)

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


feature_select_methods = {
    "no_dim_reduction": no_dim_red,
    "pca": run_pca,
    "umap": run_umap,
}
