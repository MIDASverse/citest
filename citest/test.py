import numpy as np
from scipy import stats
from sklearn.model_selection import KFold

from .classifier import CIClassifier, RFClassifier
from .data import Dataset
from .imputer import Imputer, MidasImputer
from .utils import BCEclip


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
        imputer: type[Imputer] = MidasImputer,
        classifier: type[CIClassifier] = RFClassifier,
        m: int = 10,
        n_folds: int = 10,
        classifier_args: dict = {},
        imputer_args: dict = {},
        random_state: int = 42,
        target_level: str = "variable",
        variance_method: str = "mi_crossfit",
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
        self.variance_method = variance_method

    def __repr__(self):
        return (
            f"Conditional independence test:\n"
            f"    - Data size: {self.dataset.n}\n"
            f"    - Imputer: {self.imputer}\n"
            f"    - Classifier: {self.classifier}\n"
            f"    - Folds: {self.n_folds}\n"
            f"    - Datasets: {self.m}\n"
            f"    - No. of explanatory variables: {len(self.dataset.expl_vars)}\n"
            f"    - Variance: {self.variance_method}\n"
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

        if self.variance_method not in {"legacy_fold", "mi_crossfit"}:
            raise ValueError("variance_method must be 'legacy_fold' or 'mi_crossfit'")

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

        if self.variance_method == "legacy_fold":
            self._run_legacy_fold(
                cv=cv,
                sample_idxs=sample_idxs,
                cols_idx=cols_idx,
                mask_arr=mask_arr,
                w=w,
                nfeat=nfeat,
            )
            return

        self._run_mi_crossfit(
            cv=cv,
            sample_idxs=sample_idxs,
            cols_idx=cols_idx,
            mask_arr=mask_arr,
            w=w,
            nfeat=nfeat,
        )

    def _run_legacy_fold(
        self,
        cv: KFold,
        sample_idxs: np.ndarray,
        cols_idx: list,
        mask_arr: np.ndarray,
        w: np.ndarray,
        nfeat: int,
    ) -> None:
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

                m_diffs.append(np.sum(w * (errX_mean - errXY_mean)))

            diffs.append(m_diffs)

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

    def _run_mi_crossfit(
        self,
        cv: KFold,
        sample_idxs: np.ndarray,
        cols_idx: list,
        mask_arr: np.ndarray,
        w: np.ndarray,
        nfeat: int,
    ) -> None:
        """
        MI + cross-fitting variance estimate.

        For each imputation b, build out-of-fold observation-level scores g_i^(b)
        and treat the test statistic as Q^(b) = mean_i g_i^(b). The within-imputation
        variance is estimated from the variance of fold means (with a Nadeau–Bengio
        correction) to account for within-fold dependence induced by shared model fits.
        Combine across imputations using Rubin's rules.
        """

        n = int(sample_idxs.shape[0])
        if n <= 0:
            raise ValueError("No sample indices available for testing")

        # Map global row index -> position in the sampled subset.
        pos = np.full(self.dataset.miss_data.shape[0], -1, dtype=int)
        pos[sample_idxs] = np.arange(n, dtype=int)

        m_imp = int(self.m)
        if m_imp <= 0:
            raise ValueError("m must be a positive integer")

        scores = np.full((m_imp, n), np.nan, dtype=float)
        fold_id = np.full(n, -1, dtype=int)

        w = np.asarray(w, dtype=float)

        for fold_idx, (train_rel, test_rel) in enumerate(cv.split(sample_idxs)):
            train_idx = sample_idxs[train_rel]
            test_idx = sample_idxs[test_rel]

            test_pos = pos[test_idx]
            if (test_pos < 0).any():
                raise RuntimeError("Internal index mapping failed for test indices")

            fold_id[test_pos] = fold_idx

            imputer = self.imputer(dataset=self.dataset)
            imp_datasets = imputer.get_m_complete(
                m=m_imp, train_index=train_idx, **self.imputer_args
            )
            if len(imp_datasets) != m_imp:
                raise ValueError("Imputer did not return exactly m completed datasets")

            train_R = mask_arr[train_idx]
            test_R = mask_arr[test_idx]

            # Use common random numbers across imputations within a fold so that
            # between-imputation variance (B) reflects imputation uncertainty rather than
            # classifier/permutation Monte Carlo.
            class_seed = self.rng.integers(2**32 - 1)
            perm_train = self.rng.permutation(train_idx.shape[0])
            perm_test = self.rng.permutation(test_idx.shape[0])

            for b, imp_data in enumerate(imp_datasets):
                # Check imputed data has same dimensions
                assert imp_data.shape == self.dataset.miss_data.shape
                imp_arr = imp_data.to_numpy()

                train = imp_arr[train_idx][:, cols_idx]
                test = imp_arr[test_idx][:, cols_idx]

                train_y = train[:, 0]
                test_y = test[:, 0]

                # permute outcome so CI holds but maintains symmetry
                train_u = train_y[perm_train]
                test_u = test_y[perm_test]

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

                # observation-level weighted score
                g = (errX - errXY) @ w
                scores[b, test_pos] = g

                # calculate relative score reduction (for diagnostics)
                rel_score_reduction = 1 - (errXY @ w).mean() / (errX @ w).mean()

        if np.isnan(scores).any():
            raise RuntimeError(
                "Cross-fitted scoring left NaNs; check CV splitting or index mapping."
            )

        if (fold_id < 0).any():
            raise RuntimeError(
                "Cross-fitting did not assign all observations to a fold; check CV splitting."
            )

        K = int(fold_id.max()) + 1
        if K <= 1:
            raise ValueError(
                "Need at least 2 folds for cross-fitting variance estimate"
            )

        # Within-imputation variance: use variability of fold means (with CV correction)
        # rather than Var_i(g_i)/n, since g_i are correlated within folds via shared fits.
        # i.e. within each fold, what is the average (weighted) score per imputation?
        fold_means = np.empty((m_imp, K), dtype=float)
        for k in range(K):
            idx = fold_id == k
            if not idx.any():
                raise RuntimeError("Empty fold encountered; cannot compute fold means")
            fold_means[:, k] = scores[:, idx].mean(axis=1)

        # CV estimate: equally-weighted mean over folds (matches legacy behavior)
        Q = fold_means.mean(axis=1)  # (m,)

        fold_var = fold_means.var(axis=1, ddof=1)  # (m,)

        n_test = n / K
        n_train = n - n_test
        if n_train <= 0:
            raise ValueError("Invalid CV sizes: n_train must be positive")

        cv_correction = (1.0 / K) + (n_test / n_train)  # Nadeau-Bengio correction
        U = cv_correction * fold_var

        m = float(Q.mean())
        W_bar = float(U.mean())

        if m_imp > 1:
            B = float(np.var(Q, ddof=1))
            T = W_bar + (1.0 + 1.0 / m_imp) * B
        else:
            B = 0.0
            T = W_bar

        se = np.sqrt(T) if T > 0 else 0.0
        t_k = m / se if se > 0 else 0.0

        # Barnard–Rubin style df with a conservative fallback.
        # Complete-data df: here the within-imputation variance is estimated from K fold means,
        # so the natural reference df is K-1 (as in the legacy fold-level t-test).
        df_complete = max(K - 1, 1)
        if m_imp <= 1 or B <= 0:
            df = float(df_complete)
        else:
            r = ((1.0 + 1.0 / m_imp) * B) / W_bar if W_bar > 0 else np.inf
            df_old = (m_imp - 1.0) * (1.0 + 1.0 / r) ** 2
            lambda_ = ((1.0 + 1.0 / m_imp) * B) / T if T > 0 else 0.0
            df_obs = (
                ((df_complete + 1.0) / (df_complete + 3.0))
                * df_complete
                * (1.0 - lambda_)
            )
            if not np.isfinite(df_obs) or df_obs <= 0:
                df = float(df_old)
            else:
                df = float((df_old * df_obs) / (df_old + df_obs))

        p_k = stats.t.sf(t_k, df)
        p_2s = 2 * stats.t.sf(np.abs(t_k), df)

        self.results = {
            "m": m,
            "B": B,
            "W_bar": W_bar,
            "T": T,
            "t_k": t_k,
            "p_k": p_k,
            "p_2s": p_2s,
            "df": df,
            "rel_reduction": rel_score_reduction,
        }

    def summary(self):
        """Print a summary of the test results"""

        if self.results is not None:
            df_txt = f"; df = {self.results['df']}" if "df" in self.results else ""
            print(
                f"----------------------------------------------\n"
                f"Conditional independence test results\n"
                f"----------------------------------------------\n"
                f"Outcome: {self.dataset.miss_data.columns[0]}\n"
                f"Imputer: {self.imputer}\n"
                f"Classifier: {self.classifier}\n"
                f"Variance method: {self.variance_method}\n"
                f"----------------------------------------------\n"
                f"Mean difference in BCE: {self.results['m']}\n"
                f"Test statistic: t = {self.results['t_k']}{df_txt}; p-value = {self.results['p_k']}\n"
                f"----------------------------------------------\n"
            )
        else:
            raise ValueError("Please run the test before calling summary")
