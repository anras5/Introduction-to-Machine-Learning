import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class CustomLinearRegression:

    def __init__(self, *, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.coefficient = 0
        self.intercept = 0

    def fit(self, X, y):
        y = y.to_numpy()
        if self.fit_intercept:
            # with intercept [1., ...]
            c = X.columns
            X['intercept'] = 1
            X = X[['intercept'] + c.to_list()]
            X = X.to_numpy()
            beta = (np.linalg.inv(X.T @ X) @ X.T) @ y
            self.intercept = beta[0]
            self.coefficient = beta[1:]
        else:
            # without intercept [...]
            beta = (np.linalg.inv(X.T @ X) @ X.T) @ y
            self.coefficient = beta

    def predict(self, X):
        return X @ self.coefficient + self.intercept

    def r2_score(self, y, yhat):
        return 1 - np.sum((y - yhat) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def rmse(self, y, yhat):
        return np.sqrt(np.sum((y - yhat) ** 2) / len(y))


if __name__ == "__main__":
    f1 = np.array([2.31, 7.07, 7.07, 2.18, 2.18, 2.18, 7.87, 7.87, 7.87, 7.87])
    f2 = np.array([65.2, 78.9, 61.1, 45.8, 54.2, 58.7, 96.1, 100.0, 85.9, 94.3])
    f3 = np.array([15.3, 17.8, 17.8, 18.7, 18.7, 18.7, 15.2, 15.2, 15.2, 15.2])
    y = np.array([24.0, 21.6, 34.7, 33.4, 36.2, 28.7, 27.1, 16.5, 18.9, 15.0])
    df = pd.DataFrame({"f1": f1, "f2": f2, "f3": f3, "y": y})

    lr_custom = CustomLinearRegression(fit_intercept=True)
    lr = LinearRegression(fit_intercept=True)

    lr_custom.fit(df[["f1", "f2", "f3"]], df["y"])
    lr.fit(X=df[["f1", "f2", "f3"]], y=df["y"])

    y_pred_custom = lr_custom.predict(df[["f1", "f2", "f3"]])
    y_pred = lr.predict(df[["f1", "f2", "f3"]])

    print({'Intercept': lr.intercept_ - lr_custom.intercept,
           'Coefficient': lr.coef_ - lr_custom.coefficient,
           'R2': r2_score(df["y"], y_pred) - lr_custom.r2_score(df["y"].to_numpy(), y_pred),
           'RMSE': np.sqrt(mean_squared_error(df["y"], y_pred)) - lr_custom.rmse(df["y"].to_numpy(), y_pred)})
