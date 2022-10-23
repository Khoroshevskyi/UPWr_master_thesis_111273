from typing import Dict
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.naive_bayes import GaussianNB


from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import make_scorer
import pandas as pd


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

    print(dict(clf_nn.cv_results_))
    return 0


def run_tree(x_train: pd.DataFrame, y_train: pd.DataFrame):
    tree_para = {'criterion': ['gini', 'entropy'], 'max_depth': [4, 6, 8, 10, 12, 15, 17, 20]}
    scores = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score, average='micro'),
              'recall': make_scorer(recall_score, average='micro')}

    clf_tree = GridSearchCV(DecisionTreeClassifier(random_state=15), tree_para, cv=10,
                            scoring=scores, refit='accuracy', return_train_score=True, n_jobs=4)
    clf_tree = clf_tree.fit(x_train, y_train)
    print(clf_tree.best_params_)


def run_k_neighbours(x_train: pd.DataFrame, y_train: pd.DataFrame):
    nn_para = {'n_neighbors': [3, 6, 9, 15, 30, 60, 120],
               'metric': ["euclidean", "manhattan"]}
    scores = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score, average='micro'),
              'recall': make_scorer(recall_score, average='micro')}

    clf_kn = GridSearchCV(KNeighborsClassifier(), nn_para, cv=10, scoring=scores,
                          refit='accuracy', return_train_score=True, n_jobs=4)
    clf_kn = clf_kn.fit(x_train, y_train)
    print(clf_kn.best_params_)


def run_svm(x_train: pd.DataFrame, y_train: pd.DataFrame):
    param_grid = {'penalty': ['l2'], 'C': [1, 2, 5, 10, 15, 30, 90, 320]}
    scores = {'accuracy': make_scorer(accuracy_score), 'precision': make_scorer(precision_score, average='micro'),
              'recall': make_scorer(recall_score, average='micro')}

    svc = svm.LinearSVC(random_state=10)
    clf_svc = GridSearchCV(svc, param_grid, cv=10, scoring=scores,
                           refit='accuracy', return_train_score=True, n_jobs=4)
    clf_svc.fit(x_train, y_train)


def run_bayesian(x_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, dict]:
    gnb = GaussianNB()

    # Train the model using the training sets
    gnb.fit(x_train, y_train)
    return {"hello": {"a":  1}}


def run_linear_model():
    pass


classifiers_dict = {
    "nn": run_nn_classifier,
    "tree": run_tree,
    "k_neighbours": run_k_neighbours,
    "svm": run_svm,
    "bayesian": run_bayesian,
    "linear_model": "linear model",
}