from .data import *
from .imputer import *
from .classifier import *
from .utils import BCEclip

import numpy as np
from scipy import stats

from sklearn.model_selection import KFold


class CIMissTest:
    """Run a conditional independence of missingness test on a dataset

    This class enables a test of the conditional independence of
    missingness in a dataset. It is based on a comparison of
    classifier performance with and without the outcome/target
    variable.

    Attributes:
        dataset: A Dataset object
        imputer: An Imputer object -- typically IterativeImputer
        classifier: A CIClassifier object -- typically RandomForest
        m: An integer with the number of multiply imputed datasets for the test
        n_folds: An integer with the number of folds for cross-validation
        classifier_args: A dictionary with keyword arguments for the classifier
        imputer_args: A dictionary with keyword arguments for the imputer
        results: A dictionary with the results of the test
    """

    def __init__(
        self,
        dataset: Dataset,
        imputer: Imputer = MidasImputer,
        classifier: CIClassifier = RFClassifier,
        m: int = 10,
        n_folds: int = 10,
        classifier_args: dict = {},
        imputer_args: dict = {},
        random_state: int = 42,
        target_level: str = "variable",
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
        self.target_level = target_level

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

        if self.n_folds > self.dataset.n:
            raise ValueError(
                f"Number of folds ({self.n_folds}) cannot exceed number of samples ({self.dataset.n})"
            )

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

        cols_idx = self.dataset.get_predictor_cols_idx()
        mask_arr = self.dataset.get_target_mask(level=self.target_level)
        w = self.dataset.get_target_weights(level=self.target_level)

        nfeat = len(cols_idx)

        diffs = []
        for train_idx, test_idx in cv.split(sample_idxs):

            train_idx = sample_idxs[train_idx]
            test_idx = sample_idxs[test_idx]

            imputer = self.imputer(dataset=self.dataset)
            imp_datasets = imputer.get_m_complete(
                m=self.m, train_index=train_idx, **self.imputer_args
            )

            m_diffs = []
            for imp_data in imp_datasets:
                # Check imputed data has same dimensions
                assert imp_data.shape == self.dataset.miss_data.shape
                imp_arr = imp_data.to_numpy()

                train = imp_arr[train_idx][:, cols_idx]
                test = imp_arr[test_idx][:, cols_idx]

                train_y = train[:, 0]
                test_y = test[:, 0]

                # permute outcome so CI holds but maintains symmetry
                train_u = self.rng.permutation(train_y)
                test_u = self.rng.permutation(test_y)

                train_R = mask_arr[train_idx]
                test_R = mask_arr[test_idx]

                class_seed = self.rng.integers(2**32 - 1)

                clf_kwargs = dict(self.classifier_args)
                clf_kwargs.pop("random_state", None)  # enforce rng-driven seed only

                modX = self.classifier(
                    random_state=class_seed, n_features=nfeat, **clf_kwargs
                )
                modXY = self.classifier(
                    random_state=class_seed, n_features=nfeat, **clf_kwargs
                )

                modX.fit(X=np.column_stack((train_u, train[:, 1:])), y=train_R)
                modXY.fit(X=train, y=train_R)

                predsX = modX.predict(np.column_stack((test_u, test[:, 1:])))
                predsXY = modXY.predict(test)

                assert predsX.shape == test_R.shape
                assert predsXY.shape == test_R.shape

                errX = BCEclip(predsX, test_R)
                errXY = BCEclip(predsXY, test_R)

                errX_mean = errX.mean(axis=0)
                errXY_mean = errXY.mean(axis=0)

                # PREVIOUS CODE
                # xij = np.mean(errX) - np.mean(errXY)
                # m_diffs.append(xij)

                # NEW WEIGHTED CODE
                m_diffs.append(np.sum(w * (errX_mean - errXY_mean)))

            diffs.append(m_diffs)

        # m = np.mean(np.concatenate(diffs))

        # sigma2_k = np.var([np.mean(d) for d in diffs], ddof=1)

        # n = len(sample_idxs)
        # n_per_fold = n / self.n_folds

        # F_k = self.n_folds
        # if m != 0:
        #     t_k = m / np.sqrt((1 / F_k + n_per_fold / (n - n_per_fold)) * sigma2_k)
        # else:
        #     t_k = 0.0

        # p_k = stats.t.sf(t_k, F_k - 1)

        # p_2s = 2 * stats.t.sf(np.abs(t_k), F_k - 1)

        # self.results = {
        #     "m": m,
        #     "sigma2_k": sigma2_k,
        #     "t_k": t_k,
        #     "p_k": p_k,
        #     "p_2s": p_2s,
        # }

        f_means = np.array(
            [np.mean(d) for d in diffs], dtype=float
        )  # average over imputations within fold
        f_vars = np.array(
            [np.var(d, ddof=1) if len(d) > 1 else 0.0 for d in diffs], dtype=float
        )  # variance over imputations within fold

        F_k = len(diffs)
        m_imp = self.m
        assert len(diffs[0]) == m_imp

        m = float(np.mean(f_means))  # grand mean

        B = (
            float(np.var(f_means, ddof=1)) if F_k > 1 else 0.0
        )  # between variance (Rubin's)
        W_bar = float(np.mean(f_vars)) if F_k > 0 else 0.0  # within variance (Rubin's)
        T = B + (W_bar / m_imp if m_imp > 0 else 0.0)  # total variance

        n = len(sample_idxs)
        n_per_fold = n / self.n_folds

        if n <= n_per_fold:
            raise ValueError("Invalid CV sizes: n must be > n_per_fold")

        cv_correction = (1.0 / F_k) + (
            n_per_fold / (n - n_per_fold)
        )  # Nadeau-Bengio correction

        se = np.sqrt(cv_correction * T) if T > 0 else 0.0
        t_k = m / se if se > 0 else 0.0
        p_k = stats.t.sf(t_k, F_k - 1)
        p_2s = 2 * stats.t.sf(np.abs(t_k), F_k - 1)

        self.results = {
            "m": m,
            "B": B,
            "W_bar": W_bar,
            "T": T,
            "t_k": t_k,
            "p_k": p_k,
            "p_2s": p_2s,
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
                f"----------------------------------------------\n"
            )
        else:
            raise ValueError("Please run the test before calling summary")
