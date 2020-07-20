import pandas as pd
import numpy as np

from sklearn import ensemble, metrics, model_selection
from functools import partial
from skopt import space, gp_minimize


def objective(params, param_names, X, y):
    """The objective function that is to be minimized.

    Args:
        params (list): list of parameter values
        param_names (string): list of parameter names
        X (list): input data
        y (list): target variable

    Returns:
        float: inverted mean accuracy over all folds
    """
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    folds = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []

    for idx in folds.split(X=X, y=y):
        train_idx, test_idx = idx[0], idx[1]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        acc = metrics.accuracy_score(y_test, predictions)
        accuracies.append(acc)

    # we minimize the objective function
    return np.mean(accuracies) * -1.0


if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")

    # use all features other than the target variable
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 600, name="n_estimators"),
        space.Categorical(["entropy", "gini"], name="criterion"),
        space.Real(0.01, 1, prior="uniform", name="max_features")
    ]

    optimization_function = partial(
        objective,
        param_names=param_names,
        X=X,
        y=y
    )

    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=5,
        verbose=1
    )

    print(dict(zip(param_names, result.x)))
