import pandas as pd
import numpy as np

from sklearn import ensemble, metrics, model_selection, decomposition, preprocessing, pipeline


if __name__ == "__main__":
    df = pd.read_csv("../data/train.csv")

    # use all features other than the target variable
    X = df.drop("price_range", axis=1).values
    y = df.price_range.values

    scl = preprocessing.StandardScaler()
    pca = decomposition.PCA()
    clf = ensemble.RandomForestClassifier(n_jobs=-1)

    pipe = pipeline.Pipeline([("scaling", scl), ("pca", pca), ("clf", clf)])

    param_grid = {
        "pca__n_components": [5, 10, 100, 200, 500],
        "clf__n_estimators": [100, 200, 300, 400, 500],
        "clf__max_depth": [1, 5, 10, 15],
        "clf__criterion": ["entropy", "gini"]
    }

    model = model_selection.GridSearchCV(
        estimator=pipe,
        scoring="accuracy",
        param_grid=param_grid,
        verbose=1,
        n_jobs=-1,
        cv=5
    )
    model.fit(X, y)

    print(model.best_score_, "\n", model.best_estimator_.get_params())
