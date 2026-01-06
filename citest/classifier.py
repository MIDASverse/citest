from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
import numpy as np


class NotFittedError(Exception):
    pass


class CIClassifier:
    """Base class following scikitlearn-style classifier syntax

    Plausibly any classifier can be used with this API, so long as
    two methods are defined:

        1. _fit(X,y, **kwargs) -- this must fit the data taking in an array-like
        object X and an array-like object y of the same dimensions (a missingness indicator)
        2. _predict(X) -- this must return an array of shape X.shape

    The two public methods fit() and predict() should not be altered in any subclass.

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
    """
    More specific template classifier performing probabilistic classification per-target column

    Per-target probabilistic classifier. Set target_n_jobs > 1 to parallelize fits/preds across target columns; when doing so, keep the wrapped
    estimator's n_jobs=1 to avoid oversubscription.

    Estimator must support predict_proba; constant targets are short-circuited.

    Note: n_features allows passing feature count if useful, but can be ignored.
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
    """
    Random Forest classifier

    RandomForest wrapper using outer target_n_jobs for parallel targets and inner n_jobs for per-tree parallelism. Use
    either outer parallelism (target_n_jobs > 1, n_jobs=1) or inner parallelism (target_n_jobs=1, n_jobs as desired), but avoid setting both >1.
    """

    def __init__(
        self,
        n_estimators=100,
        max_features="auto",
        min_samples_leaf=5,
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


class ETClassifier(ProbClassifier):
    """
    Extremely randomized trees classifier

    ExtraTrees wrapper using outer target_n_jobs for parallel targets and inner n_jobs for per-tree parallelism. Use
    either outer parallelism (target_n_jobs > 1, n_jobs=1) or inner parallelism (target_n_jobs=1, n_jobs as desired), but avoid setting both >1.
    """

    def __init__(
        self,
        n_estimators=100,
        max_features=None,
        min_samples_leaf=5,
        class_weight="balanced",
        n_features=None,
        target_n_jobs=1,
        n_jobs=None,
        random_state=None,
        **kwargs,
    ):
        tree_n_jobs = 1 if target_n_jobs not in (None, 1) and n_jobs is None else n_jobs
        super().__init__(
            estimator=ExtraTreesClassifier,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_features=n_features,  # ignored
            target_n_jobs=target_n_jobs,
            n_jobs=tree_n_jobs,
            random_state=random_state,
            **kwargs,
        )


class LogisticClassifier(ProbClassifier):
    """
    Logistic regression classifier

    Logistic regression per target; parallelize across targets with target_n_jobs. Constant columns are handled by returning the observed
    constant probability.
    """

    def __init__(
        self,
        penalty="l2",
        C=1e6,
        solver="liblinear",
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


### GRAVEYARD ###

# class LogisticClassifier(CIClassifier):
#     """
#     Logistic regression classifier
#     """

#     def __init__(self, **kwargs):
#         super().__init__()
#         self.logit_kwargs = kwargs
#         self.estimators_ = None
#         self.const_probs_ = None

#     def _fit(self, X, y):

#         y_arr = np.asarray(y)
#         if y_arr.ndim == 1:
#             y_arr = y_arr[:, None]

#         n_targets = y_arr.shape[1]
#         self.estimators_ = []
#         self.const_probs_ = []

#         for j in range(n_targets):
#             y_j = y_arr[:, j]
#             classes = np.unique(y_j)

#             if classes.size < 2:  # prevent fit failure
#                 self.estimators_.append(None)
#                 self.const_probs_.append(float(classes[0]))
#             else:
#                 # Fit a standard logistic regression for this column
#                 est = LogisticRegression(
#                     penalty="l2",
#                     C=1e6,
#                     solver="liblinear",
#                     max_iter=5000,
#                     **self.logit_kwargs,
#                 )
#                 est.fit(X, y_j)
#                 self.estimators_.append(est)
#                 self.const_probs_.append(None)

#     def _predict(self, X):
#         probs = []
#         for est, const_p in zip(self.estimators_, self.const_probs_):
#             if est is None:
#                 # Column was constant in training; use stored constant prob
#                 probs.append(np.full(X.shape[0], const_p))
#             else:
#                 p = est.predict_proba(X)
#                 # binary logit: p has shape (n_samples, 2), [:, 1] is P(R=1)
#                 probs.append(p[:, 1])

#         return np.column_stack(probs)


# class RandomForest(CIClassifier):
#     """
#     Random Forest classifier
#     """

#     def __init__(self, **kwargs):
#         super().__init__()

#         # add lower than sklearn default for n_estimators
#         if "n_estimators" in kwargs:
#             n_estimators = kwargs.pop("n_estimators")
#         else:
#             n_estimators = 20

#         self.model = RandomForestClassifier(
#             **kwargs, n_estimators=n_estimators, max_features=None
#         )

#     def _fit(self, X, y):

#         self.model.fit(X, y)

#     def _predict(self, X):
#         probas = self.model.predict_proba(X)
#         prob_list = probas if isinstance(probas, list) else [probas]

#         cols = [
#             (
#                 p[:, 1] if p.ndim > 1 and p.shape[1] > 1 else p[:, 0]
#             )  # probability of R = 1
#             for p in prob_list
#         ]

#         return np.column_stack(cols)
