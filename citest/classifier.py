from sklearn.ensemble import RandomForestClassifier
import numpy as np


class CIClassifier:
    """Light wrapper around scikitlearn-style classifier API

    Plausibly any classifier can be used with this API, so long as
    two methods are defined:

        1. _fit(X,y, **kwargs) -- this must fit the data taking in an array-liek
        object X and a 1d array-like object y
        2. _predict(X) -- this must return a 1d array-like object of predicted
        *probabilities* of being observed

    The two public methods fit and predict should not be altered in any subclass.

    """

    def __init__(self):
        model = None

    def _fit(self, X, y) -> None:
        """Hidden method to fit specific classifier model"""
        pass

    def _predict(self, X) -> np.ndarray:
        """Hidden method to return predictions"""
        pass

    def fit(self, X, y) -> None:
        """Fit the model"""
        self._fit(X, y)
        return None

    def predict(self, X) -> np.ndarray:
        """Predict the missing indicator"""
        return self._predict(X)


class RandomForest(CIClassifier):
    """
    Random Forest classifier
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.model = RandomForestClassifier(**kwargs)

    def _fit(self, X, y):

        self.model.fit(X, y)

    def _predict(self, X):
        return np.array(
            [
                pred[:, 1] if pred.shape[1] > 1 else pred[:, 0]  # probability of R = 1
                for pred in self.model.predict_proba(X)
            ]
        )
