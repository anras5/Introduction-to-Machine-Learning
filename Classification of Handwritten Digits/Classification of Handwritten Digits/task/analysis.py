import keras.datasets.mnist
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import GridSearchCV

# data import
(x_train, y_train), _ = keras.datasets.mnist.load_data()

# data preprocessing
n = len(x_train)
m = len(x_train[0])
x_train_reshaped = x_train.reshape((n, m * m))
x_train_reshaped = x_train_reshaped[:6000]
y_train = y_train[:6000]

# train test split
x_train, x_test, y_train, y_test = train_test_split(x_train_reshaped, y_train, test_size=0.3, random_state=40)

# normalize data
normalizer = Normalizer()
x_train_norm = normalizer.fit_transform(x_train)
x_test_norm = normalizer.fit_transform(x_test)

params_knn = {
    'n_neighbors': [3, 4],
    'weights': ['uniform', 'distance'],
    'algorithm': ['auto', 'brute']
}

params_forest = {
    'n_estimators': [300, 500],
    'max_features': ['auto', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}


def fit_predict_eval(model, name, features_train, features_test, target_train, target_test):
    # here you fit the model
    model.fit(features_train, target_train)
    # make a prediction
    y_pred = model.predict(features_test)
    # calculate accuracy and save it to score
    score = accuracy_score(y_pred, target_test)
    print(f'Model: {model}\nAccuracy: {score}\n')


# example
# code
# fit_predict_eval(
#     model=KNeighborsClassifier(),
#     name='KNeighborsClassifier',
#     features_train=x_train_norm,
#     features_test=x_test_norm,
#     target_train=y_train,
#     target_test=y_test
# )
#
# fit_predict_eval(
#     model=RandomForestClassifier(random_state=40, n_jobs=-1),
#     name='RandomForestClassifier',
#     features_train=x_train_norm,
#     features_test=x_test_norm,
#     target_train=y_train,
#     target_test=y_test
# )

# grid search knn
# grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid=params_knn, n_jobs=-1, scoring='accuracy')
# grid_search_knn.fit(x_train_norm, y_train)
# print(grid_search_knn.best_estimator_)


# grid search forest
# grid_search_forest = GridSearchCV(RandomForestClassifier(random_state=40), param_grid=params_forest,
#                                   n_jobs=-1, scoring='accuracy')
# grid_search_forest.fit(x_train_norm, y_train)
# print(grid_search_forest.best_estimator_)


print("K-nearest neighbours algorithm")
print("best estimator: KNeighborsClassifier(n_neighbors=4, weights='distance')")
knn = KNeighborsClassifier(n_neighbors=4, weights='distance')
knn.fit(x_train_norm, y_train)
y_pred_knn = knn.predict(x_test_norm)
print(f"accuracy: {accuracy_score(y_pred_knn, y_test)}\n")

print("Random forest algorithm")
print("best estimator: RandomForestClassifier(class_weight='balanced_subsample', max_features='log2',"
      " n_estimators=300, random_state=40)")
forest = RandomForestClassifier(class_weight='balanced_subsample', max_features='log2',
                                n_estimators=300, random_state=40)
forest.fit(x_train_norm, y_train)
y_pred_f = forest.predict(x_test_norm)
print(f"accuracy: {accuracy_score(y_pred_f, y_test)}")
