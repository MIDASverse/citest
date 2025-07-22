from .data import *
from .imputer import *
from .classifier import *
from .utils import BCEclip

import numpy as np
from scipy import stats

from sklearn.model_selection import KFold


class MITestRubin:
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
        m: An integer with the number of multiply imputed datasets for the test
        classifier_args: A dictionary with keyword arguments for the classifier
        imputer_args: A dictionary with keyword arguments for the imputer
        results: A dictionary with the results of the test
    """

    def __init__(
        self,
        dataset: Dataset,
        imputer: Imputer = MidasImputer,
        classifier: CIClassifier = RandomForest,
        n_folds: int = 10,
        m: int = 10,
        classifier_args: dict = {},
        imputer_args: dict = {},
        random_state: int = 42,
    ):
        self.dataset = dataset
        self.imputer = imputer
        self.classifier = classifier
        self.n_folds = n_folds
        self.m = m
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
            f"    - Datasets: {self.m}\n"
            f"    - No. of explanatory variables: {len(self.dataset.expl_vars)}\n"
        )

    def _get_cv(self):
        return KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.rng.integers(2**32 - 1),
        )

    def run(self):
        """Run the conditional independence test

        Having declared the test object, this method will run the test

        """

        cv = self._get_cv()

        if self.dataset.miss_data.shape[0] > 2000:
            sample_idxs = self.rng.choice(
                self.dataset.miss_data.shape[0], size=2000, replace=False
            )
        else:
            sample_idxs = np.arange(self.dataset.miss_data.shape[0])

        diffs = []
        vars = []
        for train_idx, test_idx in cv.split(sample_idxs):

            train_idx = sample_idxs[train_idx]
            test_idx = sample_idxs[test_idx]

            imputer = self.imputer(dataset=self.dataset)
            imp_datasets = imputer.get_m_complete(
                m=self.m, train_index=train_idx, **self.imputer_args
            )
            m_diffs = []
            m_Us = []
            cols_idx = [0] + self.dataset._expl_vars
            for imp_data in imp_datasets:
                # Check imputed data has same dimensions
                assert imp_data.shape == self.dataset.miss_data.shape

                train = imp_data.iloc[train_idx, cols_idx]
                test = imp_data.iloc[test_idx, cols_idx]

                train_R = 1 * self.dataset.mask[np.ix_(train_idx, cols_idx)]
                test_R = 1 * self.dataset.mask[np.ix_(test_idx, cols_idx)]

                class_seed = self.rng.integers(2**32 - 1)
                modX = self.classifier(random_state=class_seed, **self.classifier_args)
                modXY = self.classifier(random_state=class_seed, **self.classifier_args)

                modX.fit(X=train.iloc[:, 1:], y=train_R)
                modXY.fit(X=train, y=train_R)

                predsX = modX.predict(test.iloc[:, 1:])
                predsXY = modXY.predict(test)

                errX = BCEclip(predsX.flatten(), test_R.flatten())
                errXY = BCEclip(predsXY.flatten(), test_R.flatten())

                xij = (errXY - errX).mean()
                uij = (errXY - errX).var(ddof=1) / len(errX)
                m_diffs.append(xij)
                m_Us.append(uij)

            m_mean = np.mean(m_diffs)
            m_b = np.var(m_diffs, ddof=1)
            m_u = np.mean(m_Us)

            diffs.append(m_mean)
            vars.append(m_u + (1 + 1 / self.m) * m_b)

        m = np.mean(diffs)

        sigma2_b = np.var(diffs, ddof=1)
        t_bar = np.mean(vars)
        n_per_fold = self.dataset.n / self.n_folds

        F = self.n_folds
        if m != 0:
            t_m = m / np.sqrt(
                # ((1 / F + n_per_fold / (self.dataset.n - n_per_fold)) * sigma2_b)
                sigma2_b / F
                + t_bar / self.n_folds
            )

        else:
            t_m = 0.0
        p_m = 2 * stats.t.sf(np.abs(t_m), F - 1)

        self.results = {
            "m": m,
            # fold-imputation level statistic
            "sigma2_m": sigma2_b,
            "t_m": t_m,
            "p_m": p_m,
        }

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
                f"Fold-imputation--level: t = {self.results['t_m']}; {self.results['p_m']}\n"
                f"----------------------------------------------\n"
            )
        else:
            raise ValueError("Please run the test before calling summary")


class MITest2:
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
        m: An integer with the number of multiply imputed datasets for the test
        classifier_args: A dictionary with keyword arguments for the classifier
        imputer_args: A dictionary with keyword arguments for the imputer
        results: A dictionary with the results of the test
    """

    def __init__(
        self,
        dataset: Dataset,
        imputer: Imputer = MidasImputer,
        classifier: CIClassifier = RandomForest,
        n_folds: int = 10,
        m: int = 10,
        classifier_args: dict = {},
        imputer_args: dict = {},
        random_state: int = 42,
    ):
        self.dataset = dataset
        self.imputer = imputer
        self.classifier = classifier
        self.n_folds = n_folds
        self.m = m
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
            f"    - Datasets: {self.m}\n"
            f"    - No. of explanatory variables: {len(self.dataset.expl_vars)}\n"
        )

    def _get_cv(self):
        return KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.rng.integers(2**32 - 1),
        )

    def run(self):
        """Run the conditional independence test

        Having declared the test object, this method will run the test

        """

        cv = self._get_cv()

        if self.dataset.miss_data.shape[0] > 2000:
            sample_idxs = self.rng.choice(
                self.dataset.miss_data.shape[0], size=2000, replace=False
            )
        else:
            sample_idxs = np.arange(self.dataset.miss_data.shape[0])

        diffs = []
        for train_idx, test_idx in cv.split(sample_idxs):

            train_idx = sample_idxs[train_idx]
            test_idx = sample_idxs[test_idx]

            imputer = self.imputer(dataset=self.dataset)
            imp_datasets = imputer.get_m_complete(
                m=self.m, train_index=train_idx, **self.imputer_args
            )
            m_diffs = []
            cols_idx = [0] + self.dataset._expl_vars
            for imp_data in imp_datasets:
                # Check imputed data has same dimensions
                assert imp_data.shape == self.dataset.miss_data.shape

                train = imp_data.iloc[train_idx, cols_idx]
                test = imp_data.iloc[test_idx, cols_idx]

                train_R = 1 * self.dataset.mask[np.ix_(train_idx, cols_idx)]
                test_R = 1 * self.dataset.mask[np.ix_(test_idx, cols_idx)]

                class_seed = self.rng.integers(2**32 - 1)
                modX = self.classifier(random_state=class_seed, **self.classifier_args)
                modXY = self.classifier(random_state=class_seed, **self.classifier_args)

                modX.fit(X=train.iloc[:, 1:], y=train_R)
                modXY.fit(X=train, y=train_R)

                predsX = modX.predict(test.iloc[:, 1:])
                predsXY = modXY.predict(test)

                errX = BCEclip(predsX.flatten(), test_R.flatten())
                errXY = BCEclip(predsXY.flatten(), test_R.flatten())

                xij = np.mean(errXY) - np.mean(errX)
                m_diffs.append(xij)
            diffs.append(m_diffs)

        m = np.mean(np.concatenate(diffs))

        sigma2_m = np.var(np.concatenate(diffs), ddof=1)
        sigma2_k = np.var([np.mean(d) for d in diffs], ddof=1)

        n_per_fold = self.dataset.n / self.n_folds

        F_m = self.m * self.n_folds
        F_k = self.n_folds
        if m != 0:
            t_m = m / np.sqrt(
                (1 / F_m + n_per_fold / (self.dataset.n - n_per_fold)) * sigma2_m
            )

            t_k = m / np.sqrt(
                (1 / F_k + n_per_fold / (self.dataset.n - n_per_fold)) * sigma2_k
            )

        else:
            t_m = 0.0
            t_k = 0.0
        p_m = 2 * stats.t.sf(np.abs(t_m), F_m - 1)
        p_k = 2 * stats.t.sf(np.abs(t_k), F_k - 1)

        self.results = {
            "m": m,
            # fold-imputation level statistic
            "sigma2_m": sigma2_m,
            "t_m": t_m,
            "p_m": p_m,
            # fold-level statistic
            "sigma2_k": sigma2_k,
            "t_k": t_k,
            "p_k": p_k,
        }

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
                f"Fold--level: t = {self.results['t_k']}; p-value = {self.results['p_k']}\n"
                f"Fold-imputation--level: t = {self.results['t_m']}; {self.results['p_m']}\n"
                f"----------------------------------------------\n"
            )
        else:
            raise ValueError("Please run the test before calling summary")
