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
        "n_estimators": np.arange(100, 500, 50),
        "max_depth": np.arange(1, 15, 1),
        "criterion": ["entropy", "gini"]
    }

    model = model_selection.RandomizedSearchCV(
        estimator=clf,
        scoring="accuracy",
        n_iter=15,
        param_distributions=param_grid,
        verbose=1,
        n_jobs=-1,
        cv=5
    )
    model.fit(X, y)

    print(model.best_score_, "\n", model.best_estimator_.get_params())
