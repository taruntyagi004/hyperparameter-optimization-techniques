import pandas as pd
import numpy as np

from sklearn import ensemble, metrics, model_selection
from functools import partial
import optuna


def objective(trial, X, y):
    """The objective function that is to be maximized.

    Args:
        trial (trial): a single trial in optuna
        X (list): input data
        y (list): target variable

    Returns:
        float: mean accuracy over all folds
    """
    criterion = trial.suggest_categorical("criterion", ["entropy", "gini"])
    n_estimators = trial.suggest_int("n_estimators", 100, 1500)
    max_depth = trial.suggest_int("max_Depth", 3, 15)
    max_features = trial.suggest_uniform("max_features", 0.01, 1.0)

    model = ensemble.RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        criterion=criterion
    )

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

    # we maximize the objective function
    return np.mean(accuracies)


if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")

    # use all features other than the target variable
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    optimization_function = partial(objective, X=X, y=y)

    study = optuna.create_study(direction="maximize")
    study.optimize(optimization_function, n_trials=15)
