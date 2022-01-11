from flaml.model import LRL2Classifier, BaseEstimator
from sklearn.datasets import make_classification


def test_lrl2():
    BaseEstimator.search_space(1, "")
    X, y = make_classification(100000, 1000)
    print("start")
    lr = LRL2Classifier()
    lr.predict(X)
    lr.fit(X, y, budget=1e-5)
