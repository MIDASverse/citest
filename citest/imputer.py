import warnings

import numpy as np
import pandas as pd
from MIDAS2 import model as md
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer as skII

from .data import Dataset


class Imputer:
    """Base imputer interface for CI missingness testing.

    Subclasses must set ``completed`` (via ``__init__`` or ``_complete``)
    and implement ``get_m_complete`` to return multiply-imputed datasets.
    """

    def __init__(
        self,
        dataset: Dataset,
    ):
        self.dataset = dataset
        self.model = None
        self.completed = None

    def _complete(self, **kwargs):
        """Hidden completion method

        This method should be implemented in the subclass to complete
        the missing data
        """
        # Empty generic method -- used to impute data
        pass


class CompleteImputer(Imputer):
    """Oracle imputer that returns the full (pre-amputation) data. For simulation only."""

    def __init__(self, dataset=None):
        super().__init__(dataset)
        self.completed = self.dataset.full_data

    def get_m_complete(self, m: int = 10, train_index=None, **kwargs) -> pd.DataFrame:
        """Get m completed datasets

        This method will return m completed datasets, if they have already
        been imputed, otherwise it will call the hidden completion
        method first.

        """
        # Return imputed data once set
        return [self.completed.copy() for _ in range(m)]


class NullImputer(Imputer):
    """Zero-fill imputer. For simulation/testing only."""

    def __init__(self, dataset=None):
        super().__init__(dataset)
        self.completed = self.dataset.miss_data.fillna(0)

    def get_m_complete(self, m: int = 10, train_index=None, **kwargs) -> pd.DataFrame:
        """Get m completed datasets

        This method will return m completed datasets, if they have already
        been imputed, otherwise it will call the hidden completion
        method first.

        """
        # Return imputed data once set
        return [self.completed for _ in range(m)]


class IterativeImputer(Imputer):
    """`sklearn.impute.IterativeImputer`_ wrapper with sequential Y|X imputation.

    Fits on X columns first, then imputes Y conditional on imputed X
    to avoid outcome leakage.

    .. _sklearn.impute.IterativeImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
    """

    def __init__(self, dataset=None):
        super().__init__(dataset)

    # def _complete(self, train_index=None, **kwargs):
    #     imputer = skII(sample_posterior=True, **kwargs)
    #     imputer.set_output(transform="pandas")
    #     data = self.dataset.miss_data.copy()
    #     imputer.fit(data if train_index is None else data.iloc[train_index, :])
    #     self.model = imputer

    def _complete(self, train_index=None, **kwargs):
        # set up imputers
        imputer_X = skII(**kwargs, sample_posterior=True)
        imputer_X.set_output(transform="pandas")

        # set up data
        data = self.dataset.miss_data.copy()

        if train_index is not None:
            imputer_X.fit(data.iloc[train_index, 1:].copy())
        else:
            imputer_X.fit(data.iloc[:, 1:])

        self.completed = True
        self.model = imputer_X

    # def get_m_complete(self, m: int = 10, train_index=None, **kwargs) -> pd.DataFrame:
    #     """Get m completed datasets

    #     This method will return m completed datasets, if they have already
    #     been imputed, otherwise it will call the hidden completion
    #     method first.

    #     """
    #     # Return imputed data once set
    #     if self.model is None:
    #         self._complete(train_index=train_index, **kwargs)
    #     return [self.model.transform(self.dataset.miss_data) for _ in range(m)]

    def get_m_complete(self, m: int = 10, train_index=None, **kwargs) -> pd.DataFrame:
        """Get m completed datasets

        This method will return m completed datasets, if they have already
        been imputed, otherwise it will call the hidden completion
        method first.

        """
        # Return imputed data once set
        if self.model is None:
            self._complete(train_index=train_index, **kwargs)

        imputer_X = self.model
        data = self.dataset.miss_data.copy()
        y = data.iloc[:, [0]]
        y_missing = y.isnull().any().any()

        imputations = []
        for _ in range(m):
            data_X_imp = imputer_X.transform(data.iloc[:, 1:])

            if data_X_imp.isnull().any().any():
                raise RuntimeError(
                    "X imputation left NaNs; would allow Y to leak into X."
                )

            if y_missing:
                imputer_y = skII(**kwargs, sample_posterior=True)
                imputer_y.set_output(transform="pandas")

                if train_index is not None:
                    imputer_y.fit(
                        pd.concat(
                            [
                                data.iloc[train_index, 0],
                                data_X_imp.iloc[train_index, :],
                            ],
                            axis=1,
                        )
                    )
                else:
                    imputer_y.fit(pd.concat([data.iloc[:, 0], data_X_imp], axis=1))

                imputed = imputer_y.transform(
                    pd.concat([data.iloc[:, 0], data_X_imp], axis=1)
                )
            else:
                imputed = pd.concat([y, data_X_imp], axis=1)

            imputations.append(imputed)

        return imputations


class IterativeImputer2(Imputer):
    """`sklearn.impute.IterativeImputer`_ variant with extra numerical guards.

    Prefills constant/empty columns and retries with reduced
    ``n_nearest_features`` on ``LinAlgError``.

    .. _sklearn.impute.IterativeImputer: https://scikit-learn.org/stable/modules/generated/sklearn.impute.IterativeImputer.html
    """

    def __init__(self, dataset=None):
        super().__init__(dataset)
        self._prefill_X = {}

    @staticmethod
    def _compute_prefill_map(X_train: pd.DataFrame) -> dict:
        prefill = {}
        for col in X_train.columns:
            observed = X_train[col].dropna().to_numpy()
            if observed.size == 0:
                prefill[col] = 0.0
                continue

            uniq = np.unique(observed)
            if uniq.size <= 1:
                prefill[col] = float(uniq[0])

        return prefill

    def _apply_prefill_map(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._prefill_X:
            return X

        X_filled = X.copy()
        for col, fill_value in self._prefill_X.items():
            if col in X_filled.columns:
                X_filled[col] = X_filled[col].fillna(fill_value)
        return X_filled

    @staticmethod
    def _fit_imputer(X_train: pd.DataFrame, **kwargs):
        imputer = skII(**kwargs, sample_posterior=True)
        imputer.set_output(transform="pandas")
        imputer.fit(X_train)
        return imputer

    def _complete(self, train_index=None, **kwargs):
        data = self.dataset.miss_data.copy()

        X_all = data.iloc[:, 1:].copy()
        X_train = (
            X_all.iloc[train_index, :].copy() if train_index is not None else X_all
        )

        self._prefill_X = self._compute_prefill_map(X_train)
        X_train = self._apply_prefill_map(X_train)

        try:
            imputer_X = self._fit_imputer(X_train, **kwargs)
        except np.linalg.LinAlgError as e:
            retry_kwargs = dict(kwargs)
            if "n_nearest_features" not in retry_kwargs:
                p = int(X_train.shape[1])
                if p > 1:
                    retry_kwargs["n_nearest_features"] = min(20, p - 1)

            if retry_kwargs != kwargs:
                warnings.warn(
                    "IterativeImputer2 hit LinAlgError during fit; retrying with "
                    f"n_nearest_features={retry_kwargs.get('n_nearest_features')}. "
                    f"Original error: {e}",
                    RuntimeWarning,
                )

            imputer_X = self._fit_imputer(X_train, **retry_kwargs)

        self.completed = True
        self.model = imputer_X

    def get_m_complete(self, m: int = 10, train_index=None, **kwargs) -> pd.DataFrame:
        """Get m completed datasets (robust variant)."""

        if self.model is None:
            self._complete(train_index=train_index, **kwargs)

        imputer_X = self.model
        data = self.dataset.miss_data.copy()
        y = data.iloc[:, [0]]
        y_missing = y.isnull().any().any()

        imputations = []
        for _ in range(m):
            X_all = self._apply_prefill_map(data.iloc[:, 1:])
            data_X_imp = imputer_X.transform(X_all)

            if data_X_imp.isnull().any().any():
                raise RuntimeError(
                    "X imputation left NaNs; would allow Y to leak into X."
                )

            if y_missing:
                fit_df = pd.concat([data.iloc[:, 0], data_X_imp], axis=1)
                fit_train = (
                    fit_df.iloc[train_index, :] if train_index is not None else fit_df
                )

                try:
                    imputer_y = self._fit_imputer(fit_train, **kwargs)
                except np.linalg.LinAlgError as e:
                    retry_kwargs = dict(kwargs)
                    if "n_nearest_features" not in retry_kwargs:
                        p = int(fit_train.shape[1])
                        if p > 1:
                            retry_kwargs["n_nearest_features"] = min(20, p - 1)

                    if retry_kwargs != kwargs:
                        warnings.warn(
                            "IterativeImputer2 hit LinAlgError during outcome fit; retrying with "
                            f"n_nearest_features={retry_kwargs.get('n_nearest_features')}. "
                            f"Original error: {e}",
                            RuntimeWarning,
                        )

                    imputer_y = self._fit_imputer(fit_train, **retry_kwargs)

                imputed = imputer_y.transform(fit_df)
            else:
                imputed = pd.concat([y, data_X_imp], axis=1)

            imputations.append(imputed)

        return imputations


class MidasImputer(Imputer):
    """`MIDAS2`_ deep-learning imputer (torch).

    .. _MIDAS2: https://github.com/MIDASverse/MIDASpy
    """

    def __init__(self, dataset=None):
        super().__init__(dataset)

    def _complete(self, train_index=None, **kwargs):
        # allow manual epochs
        if "epochs" in kwargs:
            epochs = kwargs.pop("epochs")
        else:
            epochs = 250

        if "omit_first" in kwargs:
            omit_first = kwargs.pop("omit_first")
        else:
            omit_first = True

        midas_model = md.MIDAS(**kwargs)

        midas_model.fit(
            (
                self.dataset.miss_data.iloc[train_index, :].copy()
                if train_index is not None
                else self.dataset.miss_data
            ),
            epochs=epochs,
            omit_first=omit_first,
            verbose=False,
        )

        self.completed = True  # sum(imps) / len(imps)
        self.model = midas_model

    def get_m_complete(self, m: int = 10, train_index=None, **kwargs) -> pd.DataFrame:
        """Get m completed datasets

        This method will return m completed datasets, if they have already
        been imputed, otherwise it will call the hidden completion
        method first.

        """
        # Return imputed data once set
        if self.model is None:
            self._complete(train_index=train_index, **kwargs)

        if train_index is not None:
            return list(
                # pass through entire dataset as test will subset out test and train subsets
                self.model.transform(X=self.dataset.miss_data, m=m, format_X=True)
            )
        else:
            return list(self.model.transform(m=m))
