import sklearn.datasets
import sklearn.model_selection
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class CustomLogisticRegression:

    def __init__(self, fit_intercept=True, l_rate=0.01, n_epoch=100):
        self.fit_intercept = fit_intercept
        self.l_rate = l_rate
        self.n_epoch = n_epoch
        if self.fit_intercept:
            self.coef_ = [0 for _ in range(X_train.shape[1] + 1)]
        else:
            self.coef_ = [0 for _ in range(X_train.shape[1])]
        self.errors = []

    def sigmoid(self, t):
        return round(1 / (1 + np.e ** (-t)), 5)

    def predict_proba(self, row, coef_):
        t = np.sum(row * coef_[1:]) + coef_[0]
        return self.sigmoid(t)

    def fit_mse(self, X_train, y_train):
        N = len(X_train)
        for _ in range(self.n_epoch):
            error = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                if self.fit_intercept:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat)
                    for b_n in range(1, len(self.coef_)):
                        self.coef_[b_n] = self.coef_[b_n] - self.l_rate * (y_hat - y_train[i]) * y_hat * (1 - y_hat) * \
                                          row[b_n - 1]
                else:
                    for b_n in range(len(self.coef_)):
                        self.coef_[b_n] = self.coef_[b_n] - self.l_rate * (y_hat - y_train[i]) * (1 - y_hat) * row[b_n]
                error.append((y_hat - y_train[i]) ** 2)
            # error = error / N
            self.errors.append(error)

    def fit_log_loss(self, X_train, y_train):
        N = len(X_train)
        for _ in range(self.n_epoch):
            error = []
            for i, row in enumerate(X_train):
                y_hat = self.predict_proba(row, self.coef_)
                if self.fit_intercept:
                    self.coef_[0] = self.coef_[0] - self.l_rate * (y_hat - y_train[i]) / N
                    for b_n in range(1, len(self.coef_)):
                        self.coef_[b_n] = self.coef_[b_n] - self.l_rate * (y_hat - y_train[i]) * row[b_n - 1] / N
                else:
                    for b_n in range(len(self.coef_)):
                        self.coef_[b_n] = self.coef_[b_n] - self.l_rate * (y_hat - y_train[i]) * row[b_n - 1] / N
                error.append(y_train[i] * np.log(y_hat) + (1 - y_train[i]) * np.log(1 - y_hat))
            # error = error / -N
            self.errors.append(error)

    def predict(self, X_test, cut_off=0.5):
        predictions = []
        for row in X_test:
            y_hat = self.predict_proba(row, self.coef_)
            predictions.append(0 if y_hat < cut_off else 1)
        return predictions  # predictions are binary values - 0 or 1


if __name__ == '__main__':
    # load the data
    dff = sklearn.datasets.load_breast_cancer(as_frame=True).frame
    X = dff[['worst concave points', 'worst perimeter', 'worst radius']]
    y = dff['target']

    # standarize X
    scaler = sklearn.preprocessing.StandardScaler()
    X_standarized = scaler.fit_transform(X)

    # split the dataset into training and test sets
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_standarized, y.to_numpy(),
                                                                                train_size=0.8, random_state=43)

    # create custom_logistic_regression
    custom_logistic_regression_mse = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    custom_logistic_regression_log = CustomLogisticRegression(fit_intercept=True, l_rate=0.01, n_epoch=1000)
    logistic_regression = LogisticRegression(fit_intercept=True)

    # mse
    custom_logistic_regression_mse.fit_log_loss(X_train, y_train)
    y_hat = custom_logistic_regression_mse.predict(X_test)
    acc_score_mse = accuracy_score(y_test, y_hat)
    coef_mse = custom_logistic_regression_mse.coef_

    # log
    custom_logistic_regression_log.fit_log_loss(X_train, y_train)
    y_hat = custom_logistic_regression_log.predict(X_test)
    acc_score_log = accuracy_score(y_test, y_hat)
    coef_log = custom_logistic_regression_log.coef_

    # sklearn
    logistic_regression.fit(X_train, y_train)
    y_hat = logistic_regression.predict(X_test)
    acc_score_sklearn = accuracy_score(y_test, y_hat)
    coef_sklearn = logistic_regression.coef_

    print({
        'mse_accuracy': acc_score_mse,
        'logloss_accuracy': acc_score_log,
        'sklearn_accuracy': acc_score_sklearn,
        'mse_error_first': custom_logistic_regression_mse.errors[0],
        'mse_error_last': custom_logistic_regression_mse.errors[-1],
        'logloss_error_first': custom_logistic_regression_log.errors[0],
        'logloss_error_last': custom_logistic_regression_log.errors[-1]
    })

    print("""
Answers to the questions:
1) 0.00004
2) 0.00000
3) 0.00153
4) 0.00580
5) expanded
6) expanded
    """)
