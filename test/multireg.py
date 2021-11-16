from flaml import AutoML
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

# create regression data
X, y = make_regression(n_targets=3)

# split into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# train the model
model = MultiOutputRegressor(AutoML(task="regression", time_budget=60))
model.fit(X_train, y_train)

# predict
print(model.predict(X_test))