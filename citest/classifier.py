import math

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class NotFittedError(Exception):
    pass


class CIClassifier:
    """Base classifier interface for CI missingness testing.

    Subclasses must implement ``_fit(X, y)`` and ``_predict(X)``.
    The public ``fit`` and ``predict`` methods should not be overridden.
    """

    def __init__(self):
        pass

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


class ProbClassifier(CIClassifier):
    """Per-column probabilistic classifier wrapper.

    Fits a separate estimator (must support ``predict_proba``) to each
    target column. Constant columns are short-circuited.

    Set ``target_n_jobs > 1`` to parallelise across targets (keep the
    wrapped estimator's ``n_jobs=1`` to avoid oversubscription).
    """

    def __init__(
        self,
        estimator=None,
        n_features=None,
        target_n_jobs=1,
        require_proba=True,
        **kwargs,
    ):
        super().__init__()
        if not callable(estimator):
            raise TypeError("Estimator must be a callable")
        else:
            self.estimator = estimator
        self.require_proba = require_proba
        self.models_ = None
        self.const_probs_ = None
        self.n_targets_ = None
        self.n_features = n_features
        self.target_n_jobs = target_n_jobs
        self.base_kwargs = dict(
            **kwargs,
        )

    def _new_estimator(self):
        est = self.estimator(**self.base_kwargs)
        if self.require_proba and not hasattr(est, "predict_proba"):
            raise ValueError("Estimator must implement predict_proba")
        return est

    def _fit(self, X, y):
        """
        X: (n_samples, n_features)
        y: (n_samples, n_outputs) 0/1 missingness indicators
        """
        y_arr = np.asarray(y)
        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]

        self.n_targets_ = y_arr.shape[1]
        self.models_ = []
        self.const_probs_ = []

        def fit_one(y_col):
            if y_col.min() == y_col.max():  # faster constant check
                return None, float(y_col[0])
            est = self._new_estimator()
            est.fit(X, y_col)
            return est, None

        results = Parallel(n_jobs=self.target_n_jobs, prefer="threads")(
            delayed(fit_one)(y_col) for y_col in y_arr.T
        )
        self.models_, self.const_probs_ = map(list, zip(*results))

    def _predict(self, X):
        """
        Returns an (n_samples, n_outputs) array of predicted probabilities
        P(R_j = 1 | X) for each column j, matching the shape of the mask.
        """

        if self.models_ is None or self.const_probs_ is None:
            raise NotFittedError(
                "No model fits detected--please ensure fit() has been called"
            )

        probs = np.empty((X.shape[0], self.n_targets_))

        for idx, (model, const_p) in enumerate(zip(self.models_, self.const_probs_)):
            if model is None:
                probs[:, idx] = const_p
            else:
                model_proba = model.predict_proba(X)

                if hasattr(model, "classes_"):
                    # pick the column that corresponds to the observed class label (1) if present
                    class_one_idx = np.where(model.classes_ == 1)[0]
                    if class_one_idx.size > 0:
                        probs[:, idx] = model_proba[:, class_one_idx[0]]
                        continue

                # fall back to assuming observed class is in the second column
                probs[:, idx] = model_proba[:, 1]

        return probs


class RFClassifier(ProbClassifier):
    """`sklearn.ensemble.RandomForestClassifier`_ wrapper for CI testing.

    Uses piecewise ``max_features`` heuristics based on ``n_features``.
    Set ``min_samples_leaf='auto'`` for adaptive leaf sizing.

    .. _sklearn.ensemble.RandomForestClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="auto",
        min_samples_leaf="auto",
        class_weight="balanced",
        n_features=None,
        target_n_jobs=1,
        n_jobs=None,
        random_state=None,
        **kwargs,
    ):
        # set inner parallelism
        tree_n_jobs = 1 if target_n_jobs not in (None, 1) and n_jobs is None else n_jobs

        # piecewise max_features based on n_features
        if max_features == "auto":
            if n_features is None:
                max_features = (
                    "sqrt"  # default sklearn behavior when n_features unknown
                )
                raise UserWarning(
                    "n_features is None; setting max_features='sqrt' by default"
                )
            elif n_features <= 12:
                max_features = None
            elif n_features <= 80:
                max_features = 12
            else:
                max_features = "sqrt"

        super().__init__(
            estimator=RandomForestClassifier,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_features=n_features,
            target_n_jobs=target_n_jobs,
            n_jobs=tree_n_jobs,
            random_state=random_state,
            **kwargs,
        )

    @staticmethod
    def _auto_min_samples_leaf(n, n_features, lo=10, hi=50, default=5):
        if n <= 0:
            raise ValueError("n must be positive for auto min_samples_leaf")
        if n == 1:
            return 1

        if n_features > 12:
            leaf = int(default)
        else:
            leaf = int(round(math.sqrt(n)))
            leaf = max(lo, min(hi, leaf))

        upper = max(1, n // 2)
        return max(1, min(leaf, upper))

    def _fit(self, X, y):
        # adjust min_samples_leaf if set to 'auto'
        if self.base_kwargs.get("min_samples_leaf", None) == "auto":
            n = X.shape[0]
            n_features = self.n_features if self.n_features is not None else X.shape[1]
            auto_msl = self._auto_min_samples_leaf(n, n_features)
            self.base_kwargs["min_samples_leaf"] = auto_msl

        super()._fit(X, y)


class ETClassifier(ProbClassifier):
    """`sklearn.ensemble.ExtraTreesClassifier`_ wrapper for CI testing.

    Uses piecewise ``max_features`` heuristics based on ``n_features``.
    Set ``min_samples_leaf='auto'`` for adaptive leaf sizing.

    .. _sklearn.ensemble.ExtraTreesClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="auto",
        min_samples_leaf="auto",
        class_weight="balanced",
        n_features=None,
        target_n_jobs=1,
        n_jobs=None,
        random_state=None,
        **kwargs,
    ):
        tree_n_jobs = 1 if target_n_jobs not in (None, 1) and n_jobs is None else n_jobs

        if max_features == "auto":
            if n_features is None:
                max_features = "sqrt"
                raise UserWarning(
                    "n_features is None; setting max_features='sqrt' by default"
                )
            elif n_features <= 12:
                max_features = None
            elif n_features <= 80:
                max_features = 12
            else:
                max_features = "sqrt"

        super().__init__(
            estimator=ExtraTreesClassifier,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_features=n_features,
            target_n_jobs=target_n_jobs,
            n_jobs=tree_n_jobs,
            random_state=random_state,
            **kwargs,
        )

    @staticmethod
    def _auto_min_samples_leaf(n, n_features, lo=10, hi=50, default=5):
        if n <= 0:
            raise ValueError("n must be positive for auto min_samples_leaf")
        if n == 1:
            return 1

        if n_features > 12:
            leaf = int(default)
        else:
            leaf = int(round(math.sqrt(n)))
            leaf = max(lo, min(hi, leaf))

        upper = max(1, n // 2)
        return max(1, min(leaf, upper))

    def _fit(self, X, y):
        if self.base_kwargs.get("min_samples_leaf", None) == "auto":
            n = X.shape[0]
            n_features = self.n_features if self.n_features is not None else X.shape[1]
            auto_msl = self._auto_min_samples_leaf(n, n_features)
            self.base_kwargs["min_samples_leaf"] = auto_msl

        super()._fit(X, y)


class LogisticClassifier(ProbClassifier):
    """`sklearn.linear_model.LogisticRegression`_ wrapper for CI testing.

    .. _sklearn.linear_model.LogisticRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    def __init__(
        self,
        penalty="l2",
        C=1e6,
        solver="lbfgs",
        max_iter=5000,
        random_state=None,
        n_features=None,
        target_n_jobs=1,
        **kwargs,
    ):
        super().__init__(
            estimator=LogisticRegression,
            penalty=penalty,
            C=C,
            solver=solver,
            max_iter=max_iter,
            random_state=random_state,
            n_features=n_features,  # ignored
            target_n_jobs=target_n_jobs,
            **kwargs,
        )
