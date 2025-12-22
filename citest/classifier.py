from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np


class CIClassifier:
    """Base class following scikitlearn-style classifier syntax

    Plausibly any classifier can be used with this API, so long as
    two methods are defined:

        1. _fit(X,y, **kwargs) -- this must fit the data taking in an array-like
        object X and an array-like object y of the same dimensions (a missingness indicator)
        2. _predict(X) -- this must return an array of shape X.shape

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


class LogisticClassifier(CIClassifier):
    """
    Logistic regression classifier
    """

    def __init__(self, **kwargs):
        super().__init__()
        self.logit_kwargs = kwargs
        self.estimators_ = None
        self.const_probs_ = None

    def _fit(self, X, y):

        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]

        n_targets = y_arr.shape[1]
        self.estimators_ = []
        self.const_probs_ = []

        for j in range(n_targets):
            y_j = y_arr[:, j]
            classes = np.unique(y_j)

            if classes.size < 2:  # prevent fit failure
                self.estimators_.append(None)
                self.const_probs_.append(float(classes[0]))
            else:
                # Fit a standard logistic regression for this column
                est = LogisticRegression(
                    penalty=None,
                    solver="lbfgs",
                    max_iter=1000,
                    **self.logit_kwargs,
                )
                est.fit(X, y_j)
                self.estimators_.append(est)
                self.const_probs_.append(None)

    def _predict(self, X):
        probs = []
        for est, const_p in zip(self.estimators_, self.const_probs_):
            if est is None:
                # Column was constant in training; use stored constant prob
                probs.append(np.full(X.shape[0], const_p))
            else:
                p = est.predict_proba(X)
                # binary logit: p has shape (n_samples, 2), [:, 1] is P(R=1)
                probs.append(p[:, 1])

        return np.column_stack(probs)


class RFClassifier(CIClassifier):
    """
    Random Forest classifier
    """

    def __init__(
        self,
        n_estimators=200,
        max_features=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=None,
        random_state=None,
        **kwargs,
    ):
        super().__init__()

        self.base_kwargs = dict(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            **kwargs,
        )

        self.models_ = None  # list[RandomForestClassifier or None]
        self.const_probs_ = None  # list[float or None]

    def _fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples, n_outputs) 0/1 missingness indicators
        """
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]

        n_targets = y_arr.shape[1]
        self.models_ = []
        self.const_probs_ = []

        for j in range(n_targets):
            y_j = y_arr[:, j]
            classes = np.unique(y_j)

            if classes.size < 2:
                self.models_.append(None)
                self.const_probs_.append(float(classes[0]))
            else:
                rf = RandomForestClassifier(**self.base_kwargs)
                rf.fit(X, y_j)
                self.models_.append(rf)
                self.const_probs_.append(None)

    def _predict(self, X):
        """
        Returns an (n_samples, n_outputs) array of predicted probabilities
        P(R_j = 1 | X) for each column j, matching the shape of the mask.
        """
        probs = []

        for model, const_p in zip(self.models_, self.const_probs_):
            if model is None:
                probs.append(np.full(X.shape[0], const_p))
            else:
                p = model.predict_proba(X)
                probs.append(p[:, 1])

        return np.column_stack(probs)


class RandomForest(CIClassifier):
    """
    Random Forest classifier
    """

    def __init__(self, **kwargs):
        super().__init__()

        # add lower than sklearn default for n_estimators
        if "n_estimators" in kwargs:
            n_estimators = kwargs.pop("n_estimators")
        else:
            n_estimators = 20

        self.model = RandomForestClassifier(
            **kwargs, n_estimators=n_estimators, max_features=None
        )

    def _fit(self, X, y):

        self.model.fit(X, y)

    def _predict(self, X):
        probas = self.model.predict_proba(X)
        prob_list = probas if isinstance(probas, list) else [probas]

        cols = [
            (
                p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p[:, 0]
            )  # probability of R = 1
            for p in prob_list
        ]

        return np.column_stack(cols)
