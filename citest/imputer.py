import pandas as pd

from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer as skII
from MIDAS2 import model as md
from MIDAS2.utils import imp_mean

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

    def get_complete(self, **kwargs) -> pd.DataFrame:
        """Get imputed data

        This method will return the imputed data, if it has already
        been imputed, otherwise it will call the hidden completion
        method first.

        """
        # Return imputed data once set
        if self.completed is None:
            self._complete(**kwargs)
        return self.completed


class CompleteImputer(Imputer):
    """Impute missing data with complete cases

    This imputer fills in missing data with the full data. This is
    used for simulation purposes, and cannot be used for real data tests.

    """

    def __init__(self, dataset=None):
        super().__init__(dataset)
        self.completed = self.dataset.full_data


class NullImputer(Imputer):
    """Impute missing data with zeros

    This imputer fills in missing data with zeros. This is used for
    simulation testing, and likely will not work for real data tests.

    """

    def __init__(self, dataset=None):
        super().__init__(dataset)
        self.completed = self.dataset.miss_data.fillna(0)


class IterativeImputer(Imputer):
    """Impute missing data with an iterative imputer

    This imputer uses the sklearn IterativeImputer to fill in missing data.

    This imputer can be used with real data. Additional arguments may be
    passed to the imputation model through the imputer_args parameter in
    the test module.

    `max_iter` is an important parameter to consider when using this imputer.
    If you find that the imputer reports not converging, try increasing this
    value.
    """

    def __init__(self, dataset=None):
        super().__init__(dataset)

    def _complete(self, **kwargs):
        imputer = skII(**kwargs, sample_posterior=True)
        imputer.set_output(transform="pandas")
        imputer.fit(self.dataset.miss_data)

        # take m=10 draws for completed data
        imps = [imputer.transform(self.dataset.miss_data) for _ in range(10)]
        self.completed = sum(imps) / len(imps)

        self.model = imputer


class MidasImputer(Imputer):
    """Impute missing data with MIDAS

    This imputer uses the new torch version of MIDAS to fill in missing data.

    This imputer can be used with real data. Additional arguments may be
    passed to the imputation model through the imputer_args parameter in
    the test module.


    """

    def __init__(self, dataset=None):
        super().__init__(dataset)

    def _complete(self, **kwargs):
        midas_model = md.MIDAS(**kwargs)
        midas_model.fit(self.dataset.miss_data, epochs=20)

        self.completed = imp_mean(midas_model.transform(m=10), pandas=True)
        self.model = midas_model
