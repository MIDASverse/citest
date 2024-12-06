from .data import *
from .imputer import *
from .classifier import *
from .utils import BCEclip

import numpy as np
from scipy import stats


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
    ):
        self.dataset = dataset
        self.imputer = imputer
        self.classifier = classifier
        self.n_folds = n_folds
        self.repetitions = repetitions
        self.classifier_args = classifier_args
        self.imputer_args = imputer_args
        self.results = None

    def __repr__(self):
        return f""""
            Conditional independence test:
                - Data size: {self.dataset.n}
                - Imputer: {self.imputer}
                - Classifier: {self.classifier}
                - Folds: {self.n_folds}
                - Repetitions: {self.repetitions}
        """

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

        diffs = np.zeros(self.n_folds * self.repetitions)

        for r in range(self.repetitions):

            folds = np.array(
                [
                    x
                    for _ in range(self.dataset.n // self.n_folds)
                    for x in range(self.n_folds)
                ]
            )
            np.random.shuffle(folds)

            for k in range(self.n_folds):

                train = imp_data.iloc[folds != k, :]
                test = imp_data.iloc[folds == k, :]

                train_R = 1 * self.dataset.mask[folds != k, :]
                test_R = 1 * self.dataset.mask[folds == k, :]

                modX = self.classifier(random_state=1, **self.classifier_args)
                modXY = self.classifier(random_state=0, **self.classifier_args)

                modX.fit(X=train.iloc[:, 1:], y=train_R)
                modXY.fit(X=train, y=train_R)

                predsX = modX.predict(test.iloc[:, 1:])
                predsXY = modXY.predict(test)

                errX = BCEclip(predsX.flatten(), test_R.flatten())
                errXY = BCEclip(predsXY.flatten(), test_R.flatten())

                xij = errXY - errX
                diffs[r * self.repetitions + k] = xij

        m = np.mean(diffs)
        sigma2 = (1 / (len(diffs) - 1)) * np.sum((diffs - m) ** 2)

        n_per_fold = self.dataset.n / self.n_folds

        t = m / np.sqrt(
            ((1 / len(diffs)) + (n_per_fold / (self.dataset.n - n_per_fold))) * sigma2
        )
        p = stats.t.sf(np.abs(t), len(diffs) - 1) * 2

        self.results = {"m": m, "sigma2": sigma2, "t": t, "p": p}

    def summary(self):
        """Print a summary of the test results"""

        if self.results is not None:
            print(
                f"""
                ----------------------------------------------
                Conditional independence test results
                ----------------------------------------------
                Outcome: {self.dataset.miss_data.columns[0]}
                Imputer: {self.imputer}
                Classifier: {self.classifier}
                ----------------------------------------------
                Mean difference in BCE: {self.results["m"]}
                p-value: {self.results["p"]}
                ----------------------------------------------
                """
            )
        else:
            raise ValueError("Please run the test before calling summary")
