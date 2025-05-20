from .data import *
from .imputer import *
from .classifier import *
from .utils import BCEclip

import numpy as np
from scipy import stats

from sklearn.model_selection import RepeatedKFold


class RLTest:
    """Run a conditional independence test on a dataset

    This class forms the basis of the conditional independence test. It
    will complete the algorithm from imputing values to running the
    classifier accuracy test using repeated k-fold cross-validation (with
    bias correction.)

    Attributes:
        dataset: A Dataset object
        imputer: An Imputer object -- typically IterativeImputer
        classifier: A CIClassifier object -- typically RandomForest
        n_folds: An integer with the number of folds for cross-validation
        repetitions: An integer with the number of repetitions for the test
        classifier_args: A dictionary with keyword arguments for the classifier
        imputer_args: A dictionary with keyword arguments for the imputer
        results: A dictionary with the results of the test
    """

    def __init__(
        self,
        dataset: Dataset,
        imputer: Imputer,
        classifier: CIClassifier,
        n_folds: int = 10,
        repetitions: int = 10,
        classifier_args: dict = {},
        imputer_args: dict = {},
        random_state: int = 42,
    ):
        self.dataset = dataset
        self.imputer = imputer
        self.classifier = classifier
        self.n_folds = n_folds
        self.repetitions = repetitions
        self.classifier_args = classifier_args
        self.imputer_args = imputer_args
        self.results = None
        self.rng = np.random.default_rng(random_state)

    def __repr__(self):
        return (
            f"Conditional independence test:\n"
            f"    - Data size: {self.dataset.n}\n"
            f"    - Imputer: {self.imputer}\n"
            f"    - Classifier: {self.classifier}\n"
            f"    - Folds: {self.n_folds}\n"
            f"    - Repetitions: {self.repetitions}\n"
        )

    def _get_cv(self):
        return RepeatedKFold(
            n_splits=self.n_folds,
            n_repeats=self.repetitions,
            random_state=self.rng.integers(2**32 - 1),
        )

    def run(self):
        """Run the conditional independence test

        Having declared the test object, this method will run the test

        """

        # Impute data
        imputer = self.imputer(dataset=self.dataset)
        imp_data = imputer.get_complete(**self.imputer_args)

        ## Check imputed data has same dimensions
        assert imp_data.shape == self.dataset.miss_data.shape

        # Classifier test

        diffs = []
        cv = self._get_cv()

        for train_idx, test_idx in cv.split(np.arange(imp_data.shape[0])):

            train = imp_data.iloc[train_idx, :]
            test = imp_data.iloc[test_idx, :]

            train_R = 1 * self.dataset.mask[train_idx, :]
            test_R = 1 * self.dataset.mask[test_idx, :]

            classifier_seed = self.rng.integers(2**32 - 1)
            modX = self.classifier(
                random_state=classifier_seed, max_features=None, **self.classifier_args
            )
            modXY = self.classifier(
                random_state=classifier_seed, max_features=None, **self.classifier_args
            )

            modX.fit(X=train.iloc[:, 1:], y=train_R)
            modXY.fit(X=train, y=train_R)

            predsX = modX.predict(test.iloc[:, 1:])
            predsXY = modXY.predict(test)

            errX = BCEclip(predsX.flatten(), test_R.flatten())
            errXY = BCEclip(predsXY.flatten(), test_R.flatten())

            xij = errXY - errX
            diffs.append(xij)

        # fold-level statistic
        m = np.mean(diffs)
        sigma2 = np.var(diffs, ddof=1)
        n_per_fold = self.dataset.n / self.n_folds
        F = self.repetitions * self.n_folds
        t = m / np.sqrt((1 / F + n_per_fold / (self.dataset.n - n_per_fold)) * sigma2)
        p = 2 * stats.t.sf(np.abs(t), F - 1)
        self.results = {"m": m, "sigma2": sigma2, "t": t, "p": p}

    def summary(self):
        """Print a summary of the test results"""

        if self.results is not None:
            print(
                f"----------------------------------------------\n"
                f"Conditional independence test results\n"
                f"----------------------------------------------\n"
                f"Outcome: {self.dataset.miss_data.columns[0]}\n"
                f"Imputer: {self.imputer}\n"
                f"Classifier: {self.classifier}\n"
                f"----------------------------------------------\n"
                f"Mean difference in BCE: {self.results['m']}\n"
                f"p-value: {self.results['p']}\n"
                f"----------------------------------------------\n"
            )
        else:
            raise ValueError("Please run the test before calling summary")
