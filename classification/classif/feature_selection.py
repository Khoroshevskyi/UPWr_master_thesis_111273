"""
# Features Selection
When it comes to exploring the features of our dataset, we realize that we actually have no idea of what the quantitative variables mean; we have some indications on the general kind of data that were collected, but we are operating in a blind manner.

In this context, we deem more useful to use all the information provided by the available features trying to clean out the noise in the data by means of PCA. Indeed, by selecting a subset of the possible components to both reduce the number of features and the noise, by choosing a lower proportion of the explained variance.
"""
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel

# No feature selection
# 1. Variance Threshold
# 2. Univariate feature selection
# 3. L1-based feature selection


def no_selection(x: pd.DataFrame, y) -> pd.DataFrame:
    return x


def var_threshold(x: pd.DataFrame, y, threshold: float = 0.5) -> pd.DataFrame:
    sel = VarianceThreshold(threshold=threshold)
    X_norm_var = sel.fit_transform(x)
    return X_norm_var


# def variance_select(x: pd.DataFrame, threshold: int = 50):
#     """
#     Feature selection by variance
#     :param x: X dataframe
#     :param threshold: Variance threshold
#     :return: selected x
#     """
#     selector = VarianceThreshold(threshold)
#     selector.fit(x)
#
#     x_n = x[x.columns[selector.get_support()]]
#
#     print(f"after variance selection:")
#
#     print(x_n)
#     return x_n


def select_univariate(x: pd.DataFrame, y) -> pd.DataFrame:
    return SelectKBest(chi2, k=10).fit_transform(x, y)


def select_l1(x: pd.DataFrame, y) -> pd.DataFrame:
    lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(x, y)
    model = SelectFromModel(lsvc, prefit=True)
    X_new = model.transform(x)
    return X_new


selection_dict = {
    "no_selection": no_selection,
    "var_threshold": var_threshold,
    # "var2": variance_select,
    "select_univariate": select_univariate,
    "select_l1": select_l1,
}
