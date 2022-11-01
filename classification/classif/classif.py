from typing import Dict
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, zero_one_loss
from sklearn.metrics import classification_report

from sklearn.metrics import precision_recall_curve


def _calculate_f1(precision, recall):
    """
    Calculate F1 score
    :param precision: Precision
    :param recall: Recall
    :return: F1 score
    """
    return 2 * (precision * recall) / (precision + recall)


def run_nn_classifier(x_train: pd.DataFrame, y_train: pd.DataFrame) -> dict:
    param_grid = {'activation': ['logistic', 'relu'], 'alpha': [0.0001, 0.001]}
    scores = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='micro'),
        'recall': make_scorer(recall_score, average='micro')}

    neu_net = MLPClassifier(random_state=1, max_iter=300)
    clf_nn = GridSearchCV(neu_net, param_grid, cv=10, scoring=scores,
                          refit='accuracy', return_train_score=True, n_jobs=4)
    clf_nn.fit(x_train, y_train)

    res_df = pd.DataFrame(clf_nn.cv_results_)
    res_df1 = \
    res_df.sort_values(by='rank_test_accuracy')[['rank_test_accuracy', 'mean_test_precision', 'mean_test_recall']].iloc[
        0]
    res_df2 = res_df1.to_dict()
    res_df2["f1"] = _calculate_f1(res_df2['mean_test_precision'], res_df2['mean_test_recall'])
    return res_df2


def run_tree(x_train: pd.DataFrame, y_train: pd.DataFrame):
    tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8, 10, 12, 15, 17, 20]}
    scores = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score, average='micro'),
              'recall': make_scorer(recall_score, average='micro')}

    clf_tree = GridSearchCV(DecisionTreeClassifier(random_state=15), tree_para, cv=10,
                            scoring=scores, refit='accuracy', return_train_score=True, n_jobs=4)
    clf_tree = clf_tree.fit(x_train, y_train)
    ####
    res_df = pd.DataFrame(clf_tree.cv_results_)
    res_df1 = \
    res_df.sort_values(by='rank_test_accuracy')[['rank_test_accuracy', 'mean_test_precision', 'mean_test_recall']].iloc[
        0]
    res_df2 = res_df1.to_dict()
    res_df2["f1"] = _calculate_f1(res_df2['mean_test_precision'], res_df2['mean_test_recall'])
    return res_df2


def run_k_neighbours(x_train: pd.DataFrame, y_train: pd.DataFrame):
    nn_para = {'n_neighbors': [3, 6, 9, 15, 30, 60, 120],
               'metric': ["euclidean", "manhattan"]}
    scores = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score, average='micro'),
              'recall': make_scorer(recall_score, average='micro')}

    clf_kn = GridSearchCV(KNeighborsClassifier(), nn_para, cv=10, scoring=scores,
                          refit='accuracy', return_train_score=True, n_jobs=4)
    clf_kn = clf_kn.fit(x_train, y_train)
    ####
    res_df = pd.DataFrame(clf_kn.cv_results_)
    res_df1 = \
    res_df.sort_values(by='rank_test_accuracy')[['rank_test_accuracy', 'mean_test_precision', 'mean_test_recall']].iloc[
        0]
    res_df2 = res_df1.to_dict()
    res_df2["f1"] = _calculate_f1(res_df2['mean_test_precision'], res_df2['mean_test_recall'])
    return res_df2


def run_svm(x_train: pd.DataFrame, y_train: pd.DataFrame):
    param_grid = {'penalty': ['l2'], 'C': [1, 2, 5, 10, 15, 30, 90, 320]}
    scores = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score, average='micro'),
              'recall': make_scorer(recall_score, average='micro')}

    svc = svm.LinearSVC(random_state=10)
    clf_svc = GridSearchCV(svc, param_grid, cv=10, scoring=scores,
                           refit='accuracy', return_train_score=True, n_jobs=4)
    clf_svc.fit(x_train, y_train)
    ###
    res_df = pd.DataFrame(clf_svc.cv_results_)
    res_df1 = \
    res_df.sort_values(by='rank_test_accuracy')[['rank_test_accuracy', 'mean_test_precision', 'mean_test_recall']].iloc[
        0]
    res_df2 = res_df1.to_dict()
    res_df2["f1"] = _calculate_f1(res_df2['mean_test_precision'], res_df2['mean_test_recall'])
    return res_df2


def run_bayesian(x_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, dict]:

    X_train1, X_test1, y_train1, y_test1 = train_test_split(x_train, y_train, test_size=0.5, random_state=0)
    gnb = GaussianNB()
    y_pred = gnb.fit(X_train1, y_train1).predict(X_test1)

    f1 = f1_score(y_test1, y_pred, average="macro")
    precision = precision_score(y_test1, y_pred, average="macro")
    recall = recall_score(y_test1, y_pred, average="macro")


    res_df2 = {
        'rank_test_accuracy': 1,
        'mean_test_precision': precision,
        'mean_test_recall': recall,
        "f1": f1
    }

    return res_df2


# def run_linear_model():
#     pass


classifiers_dict = {
    "nn": run_nn_classifier,
    "tree": run_tree,
    "k_neighbours": run_k_neighbours,
    "svm": run_svm,
    "bayesian": run_bayesian,
    # "linear_model": "linear model",
}
