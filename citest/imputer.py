import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer as skII
from MIDAS2 import model as md

from .data import Dataset


class Imputer:
    """Base model for imputing missing data

    This class contains the core structure for imputing and accessing
    missing data.

    Plauibly, any imputer can be used with this API, so long as the `completed`
    attribute is set either in the intialization or by a custom _complete method
    call.

    Attributes:
        dataset: A Dataset object
        model: A fitted imputation model
        completed: A numpy array with all missing data imputed

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
    """Impute missing data with complete cases

    This imputer fills in missing data with the full data. This is
    used for simulation purposes, and cannot be used for real data tests.

    """

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
    """Impute missing data with zeros

    This imputer fills in missing data with zeros. This is used for
    simulation testing, and likely will not work for real data tests.

    """

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
    """Impute missing data with an iterative imputer

    This imputer uses the sklearn IterativeImputer to fill in missing data.

    This imputer can be used with real data. Additional arguments may be
    passed to the imputation model through the imputer_args parameter in
    the test module.

    `max_iter` is an important parameter to consider when using this imputer.
    If you find that the imputer does not converge, try increasing this
    value.
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


class MidasImputer(Imputer):
    """Impute missing data with MIDAS

    This imputer uses the new torch version of MIDAS to fill in missing data.

    This imputer can be used with real data. Additional arguments may be
    passed to the imputation model through the imputer_args parameter in
    the test module.


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
