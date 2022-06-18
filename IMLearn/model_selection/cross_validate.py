from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> \
        Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    train_score, validation_score = float(0), float(0)
    folds = np.mod(range(X.shape[0]), cv)
    for k in range(cv):
        train_X, train_y, validation_X, validation_y = \
            X[folds != k], y[folds != k], X[folds == k], y[folds == k]
        estimator.fit(train_X, train_y)
        predicted_train_y = estimator.predict(train_X)
        predicted_validation_y = estimator.predict(validation_X)
        train_score += scoring(predicted_train_y, train_y)
        validation_score += scoring(predicted_validation_y, validation_y)

    return (train_score / cv, validation_score / cv)