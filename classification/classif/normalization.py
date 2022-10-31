# This file contain scripts for data normalization
# 1. no_norm
# 2. min_max_norm
# 3. z_score_norm
# 4. robust_scaler
# 5. standard_scaler
# 6. log2_transform


import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler


def no_norm(x: pd.DataFrame) -> pd.DataFrame:
    """
    Return Unchanged dataframe
    :param x: initial dataframe
    :return: normalized dataframe (same as initial dataframe)
    """
    return x


def min_max_norm(x: pd.DataFrame) -> pd.DataFrame:
    """
    Min-max normalization
    For every feature, the minimum value of that feature gets transformed into a 0,
    the maximum value gets transformed into a 1,
    and every other value gets transformed into a decimal between 0 and 1.
    :param x: initial dataframeS
    :return: normalized dataframe
    """
    scaler = MinMaxScaler()
    scaler.fit(x)

    # return (x - x.min()) / (x.max() - x.min())
    return scaler.transform(x)

def z_score_norm(x: pd.DataFrame) -> pd.DataFrame:
    """
    Z-Score Normalization
    If a value is exactly equal to the mean of all the values of the feature, it will be normalized to 0.
    If it is below the mean, it will be a negative number, and if it is above the mean it will be a positive number.
    :param x: initial dataframe
    :return: normalized dataframe
    """
    return x.apply(lambda iterator: ((iterator - iterator.mean()) / iterator.std()).round(2))


def robust_scaler(x: pd.DataFrame) -> pd.DataFrame:
    """
    Robust Scaler
    This Scaler removes the median and scales the data according to the quantile range.
    The IQR is the range between the 1st quartile (25th quantile) and the 3rd quartile (75th quantile).
    :param x: initial dataframe
    :return: normalized dataframe
    """
    return RobustScaler().fit_transform(x)


def standard_scaler(x: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize features by removing the mean and scaling to unit variance.
    The standard score of a sample x is calculated as:
    z = (x - u) / s
    where u is the mean of the training samples or zero if with_mean=False,
    and s is the standard deviation of the training samples or one if with_std=False.
    :param x: initial dataframe
    :return: normalized dataframe
    """
    return StandardScaler().fit_transform(x)


def log2_transform(x: pd.DataFrame) -> pd.DataFrame:
    """
    The log2-median transformation is the ssn (simple scaling normalization) method in lumi.
    It takes the non-logged expression value and divides it by the ratio of its column
    (sample) median to the mean of all the sample medians.
    :param x: initial dataframe
    :return: normalized dataframe
    """
    return x.transform(lambda x1: np.log2(x1 + 1))


# One more normalization: X_norm2 = X.apply(lambda iterator: ((iterator - iterator.mean())/iterator.std()).round(2))

normal_funct = {
    "no_norm": no_norm,
    "min_max_norm": min_max_norm,
    "z_score_norm": z_score_norm,
    "robust_scaler": robust_scaler,
    "standard_scaler": standard_scaler,
    "log2_transform": log2_transform,
}
