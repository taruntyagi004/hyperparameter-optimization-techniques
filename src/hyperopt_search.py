import pandas as pd
import numpy as np

from sklearn import ensemble, metrics, model_selection
from functools import partial
from hyperopt import hp, fmin, Trials, tpe
from hyperopt.pyll.base import scope


def objective(params, X, y):
    """The objective function that is to be minimized.

    Args:
        params (list): list of parameter values
        X (list): input data
        y (list): target variable

    Returns:
        float: inverted mean accuracy over all folds
    """
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

    param_space = {
        "max_depth": scope.int(hp.quniform("max_depth", 3, 15, 1)),
        "n_estimators": scope.int(hp.quniform("n_estimators", 100, 500, 1)),
        "criterion": hp.choice("criterion", ["entropy", "gini"]),
        "max_features": hp.uniform("max_features", 0.01, 1)
    }

    optimization_function = partial(
        objective,
        X=X,
        y=y
    )

    trials = Trials()

    # minimize the optimization function
    result = fmin(
        optimization_function,
        space=param_space,
        algo=tpe.suggest,
        max_evals=15,
        trials=trials
    )

    print(result)
