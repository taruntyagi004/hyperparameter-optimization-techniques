import pandas as pd
import numpy as np

from sklearn import ensemble, metrics, model_selection


if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")

    # use all features other than the target variable
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    clf = ensemble.RandomForestClassifier(n_jobs=-1)

    param_grid = {
        "n_estimators": [100, 200, 300, 400, 500],
        "max_depth": [1, 5, 10, 15],
        "criterion": ["entropy", "gini"]
    }

    model = model_selection.GridSearchCV(
        estimator=clf,
        scoring="accuracy",
        param_grid=param_grid,
        verbose=1,
        n_jobs=-1,
        cv=5
    )
    model.fit(X, y)

    print(model.best_score_, "\n", model.best_estimator_.get_params())
