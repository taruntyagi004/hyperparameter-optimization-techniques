# Hyperparameter Optimization Techniques

In this project, we explore several techniques that can be used for hyperparameter optimization. More specifically, we explore the following techniques:

- Grid Search (with and without the usage of pipelines)
- Random Search (with and without the usage of pipelines)
- Bayesian Optimization with Gaussian Processes
- Hyperopt (<https://github.com/hyperopt/hyperopt>)
- Optuna (<https://github.com/optuna/optuna>)

We use a cleaned and balanced dataset for this project so that we can solely focus on the hyperparameter optimization. We use the [Mobile Price Classification](https://www.kaggle.com/iabhishekofficial/mobile-price-classification) dataset, which contains information about mobile devices. (See the Kaggle link for details.) Furthermore, we use a random forest model to predict the `price_range` column. This target variable can take four different values, namely 0 (low cost), 1 (medium cost), 2 (high cost), and 3 (very high cost).
