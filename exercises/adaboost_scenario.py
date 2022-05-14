import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    model = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    train_errs, test_errs = [], []
    learners = np.arange(1, n_learners)
    for iteration in learners:
        train_errs.append(model.partial_loss(train_X, train_y, iteration))
        test_errs.append(model.partial_loss(test_X, test_y, iteration))

    make_subplots() \
        .add_traces([go.Scatter(x=learners, y=test_errs, mode='lines',
                                marker=dict(color='red', line_width=1), name='Test error'),
                     go.Scatter(x=learners, y=train_errs, mode='lines',
                                marker=dict(color='green', line_width=1), name='Train error')]) \
        .update_layout(xaxis_title="Number of weak learners", yaxis_title="Error rate",
                       title="Error rate as an output to number of weak learners") \
        .show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    limits = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])
    fig = make_subplots(rows=2, cols=2, subplot_titles=[f"Number of weak learners - {n}" for n in T],
                        horizontal_spacing=0.05, vertical_spacing=.2)

    for ind, t in enumerate(T):
        fig.add_traces([decision_surface(lambda x: model.partial_predict(x, t), limits[0], limits[1],
                                         showscale=False, colorscale=["red", "blue"]),
                        go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                                   marker=dict(color=train_y, colorscale=["red", "blue"],
                                               line=dict(color="black", width=1)))],
                       rows=(ind // 2) + 1, cols=(ind % 2) + 1) \
            .update_layout(title='decision surface as an output to number of weak learners',
                           margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(visible=False)
    fig.show()

    # Question 3: Decision surface of best performing ensemble
    test_error = []
    for T in learners:
        test_error.append(model.partial_loss(test_X, test_y, T))
    best_learner_idx = np.argmin(test_errs) + 1

    make_subplots() \
        .add_traces(
        [decision_surface(lambda x: model.partial_predict(x, best_learner_idx), limits[0], limits[1],
                          showscale=False),
         go.Scatter(x=test_X[:, 0], y=test_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(color=test_y,
                                colorscale=[custom[0], custom[-1]],
                                line=dict(color="black", width=1)))]) \
        .update_layout(title=f"Decision Surface as an output to number of weak learners = {best_learner_idx}, accuracy is "
                             f"{1 - test_error[best_learner_idx]}",
                       font=dict(family="Arial", size=20),
                       margin=dict(t=50)) \
        .show()

    # Question 4: Decision surface with weighted samples
    D = model.D_ / np.max(model.D_) * 5
    make_subplots().add_traces([decision_surface(model.predict, limits[0], limits[1], showscale=False),
         go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers", showlegend=False,
                    marker=dict(colorscale=[custom[0], custom[-1]], color=train_y,
                                size=D, line=dict(color="black", width=1)))]) \
        .update_layout(margin=dict(t=100)) \
        .show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
    print("finish")
