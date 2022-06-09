from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    func = lambda x: (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)
    noise = np.random.normal(0, noise, n_samples)
    X = np.linspace(-1.2, 2, n_samples)
    y = func(X) + noise

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(X), pd.Series(y), 2.0 / 3)
    X_train, y_train, X_test, y_test = X_train[0].to_numpy(), y_train.to_numpy(), X_test[0].to_numpy(), \
                                       y_test.to_numpy()

    go.Figure([go.Scatter(x=X, y=func(X), mode='lines', name="Noiseless Model"),
               go.Scatter(x=X_test, y=y_test, mode='markers', name="Test"),
               go.Scatter(x=X_train, y=y_train, mode='markers', name="Train")]) \
        .update_layout(height=600, title_text="Noiseless Model with train and test sets") \
        .show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    deg_list = np.arange(11)
    train_arr, validation_arr = np.zeros(11), np.zeros(11)
    for deg in deg_list:
        train_arr[deg], validation_arr[deg] = cross_validate(PolynomialFitting(deg), X_train, y_train,
                                                             mean_square_error, 5)

    go.Figure([go.Scatter(x=deg_list, y=validation_arr, mode='lines', name="Validation"),
               go.Scatter(x=deg_list, y=train_arr, mode='lines', name="Train")]) \
        .update_layout(height=600, title_text="Errors for Train and Validation Sets as a Function "
                                              "of The Polynomial Degree") \
        .show()
    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    best_degree = np.argmin(validation_arr)
    model = PolynomialFitting(best_degree)
    model.fit(X_train, y_train)
    error = mean_square_error(y_test, model.predict(X_test))
    print(f"The best degree is- {best_degree}, the Test Error is- {error}")


def select_regularization_parameter(n_samples: int = 50, n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True, as_frame=True)
    train_X, train_y, test_X, test_y = split_train_test(X, y,
                                                        n_samples / X.shape[0])
    train_X, test_X, train_y, test_y = train_X.to_numpy(), test_X.to_numpy(), train_y.to_numpy(), test_y.to_numpy()
    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    ridge_train_scores, ridge_validation_scores, lasso_train_scores, lasso_validation_scores = [], [], [], []
    ridge_lam_range, lasso_lam_range = 0.001, 0.01
    ridge_lam_array = np.linspace(0, ridge_lam_range, n_evaluations).tolist()
    lasso_lam_array = np.linspace(0, lasso_lam_range,
                                  n_evaluations + 1).tolist()[1:]
    for lam in ridge_lam_array:
        lam_train_score, lam_val_score = cross_validate(RidgeRegression(lam),
                                                        train_X, train_y,
                                                        mean_square_error)
        ridge_train_scores.append(lam_train_score)
        ridge_validation_scores.append(lam_val_score)

    go.Figure() \
        .add_trace(
        go.Scatter(x=ridge_lam_array, y=ridge_train_scores,
                   mode="markers", name="Train Errors")) \
        .add_trace(
        go.Scatter(x=ridge_lam_array, y=ridge_validation_scores,
                   mode="markers", name="Validation Errors")) \
        .update_layout(
        title=f"Validation and Train Errors of Ridge Regression",
        xaxis_title="k", yaxis_title="Error",
    ) \
        .show()

    for lam in lasso_lam_array:
        lam_train_score, lam_val_score = cross_validate(
            Lasso(lam, max_iter=10000),
            train_X, train_y,
            mean_square_error)
        lasso_train_scores.append(lam_train_score)
        lasso_validation_scores.append(lam_val_score)

    go.Figure()\
        .add_trace(
        go.Scatter(x=lasso_lam_array, y=lasso_train_scores, mode="markers",name="Train Errors"))\
        .add_trace(
        go.Scatter(x=lasso_lam_array, y=lasso_validation_scores,
                   mode="markers", name="Validation Errors"))\
        .update_layout(
        title="Train and Validation Errors of Lasso Regression",
        xaxis_title="k", yaxis_title="Error")\
        .show()

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    min_lam_ridge = ridge_lam_array[np.argmin(ridge_validation_scores)]
    min_lam_lasso = lasso_lam_array[np.argmin(lasso_validation_scores)]
    best_ridge_model = RidgeRegression(min_lam_ridge).fit(train_X, train_y)
    print(
        f"Test Error for n_samples: {n_samples}\n  "
        f"Ridge Regression with lam = {min_lam_ridge} : "
        f"{best_ridge_model.loss(test_X, test_y)}")
    best_lasso = Lasso(min_lam_lasso, max_iter=10000).fit(train_X,
                                                          train_y)
    best_lasso_test_pred = best_lasso.predict(test_X)
    print(f"Test Error for n_samples: {n_samples}\n Lasso Regression with lam = {min_lam_lasso} : "
          f"{mean_square_error(best_lasso_test_pred, test_y)}")
    my_ls = LinearRegression().fit(train_X, train_y)
    print(
        f"Test Error for n_samples: {n_samples}\n  "
        f"Our LS Algorithm : "
        f"{my_ls.loss(test_X, test_y)}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree(noise=5)
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
