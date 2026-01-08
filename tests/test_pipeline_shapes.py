import unittest
import numpy as np
import pandas as pd

from citest.data import Dataset
from citest.test import CIMissTest


class EchoImputer:
    """Imputer stub that simply returns the observed data with sentinels."""

    def __init__(self, dataset):
        self.dataset = dataset

    def get_m_complete(self, m=1, train_index=None, **kwargs):
        filled = self.dataset.miss_data.fillna(-1).copy()
        return [filled.copy() for _ in range(m)]


class ShapeCheckingClassifier:
    """Classifier stub that records the shapes it sees to catch transpositions."""

    fit_records = []
    predict_records = []

    def __init__(self, random_state=None, **kwargs):
        self._target_width = None
        self._n_features = None

    def fit(self, X, y):
        # Catch common transposition mistakes
        assert X.shape[0] == y.shape[0], "Rows of X and y should align"
        self._n_features = X.shape[1]
        self._target_width = y.shape[1] if y.ndim > 1 else 1
        ShapeCheckingClassifier.fit_records.append((X.shape, y.shape))

    def predict(self, X):
        assert self._n_features == X.shape[1], "Feature axis must stay aligned"
        ShapeCheckingClassifier.predict_records.append(X.shape)
        return np.full((X.shape[0], self._target_width), 0.5)


class TestPipelineShapes(unittest.TestCase):
    def setUp(self):
        raw = pd.DataFrame(
            {
                "Y": [1, 0, 1, 0],
                "A": [0.5, np.nan, 0.1, 1.0],
                "B": [1, 2, 3, np.nan],
            }
        )
        self.dataset = Dataset()
        self.dataset.make(raw, y="Y", expl_vars=["A", "B"], _onehot=False)

    def test_mask_matches_missingness_orientation(self):
        observed_missing = np.argwhere(self.dataset.miss_data.isnull().to_numpy())
        mask_missing = np.argwhere(~self.dataset.mask)
        np.testing.assert_array_equal(observed_missing, mask_missing)

    def test_imputer_preserves_axes(self):
        imputer = EchoImputer(self.dataset)
        imputed_sets = imputer.get_m_complete(m=3)

        self.assertEqual(len(imputed_sets), 3)
        for frame in imputed_sets:
            self.assertEqual(tuple(frame.shape), tuple(self.dataset.miss_data.shape))
            self.assertTrue((frame.columns == self.dataset.miss_data.columns).all())
            self.assertTrue((frame.index == self.dataset.miss_data.index).all())

    def test_cimiss_pipeline_detects_transpose(self):
        ShapeCheckingClassifier.fit_records = []
        ShapeCheckingClassifier.predict_records = []

        test = CIMissTest(
            dataset=self.dataset,
            imputer=EchoImputer,
            classifier=ShapeCheckingClassifier,
            m=1,
            n_folds=2,
        )

        test.run()

        # X-only branch now includes a permuted outcome column, so both models should
        # see the same feature width (outcome + expl vars) on every fit
        expected_width = len(self.dataset._expl_vars) + 1
        feature_counts = [rec[0][1] for rec in ShapeCheckingClassifier.fit_records]
        self.assertEqual(set(feature_counts), {expected_width})
        self.assertEqual(len(feature_counts), test.n_folds * test.m * 2)

        self.assertTrue(
            all(
                x_shape[0] == y_shape[0]
                for x_shape, y_shape in ShapeCheckingClassifier.fit_records
            )
        )

    def test_variable_level_targets_align_with_one_hot_predictors(self):
        ShapeCheckingClassifier.fit_records = []
        ShapeCheckingClassifier.predict_records = []

        raw = pd.DataFrame(
            {
                "Y": [1, 0, 1, 0, 1, 0],
                "A": [0.5, 0.2, np.nan, 1.0, 1.2, 0.0],
                "C": ["r", "b", "r", np.nan, "g", "b"],
            }
        )
        dataset = Dataset()
        dataset.make(raw, y="Y", expl_vars=["A", "C"])

        test = CIMissTest(
            dataset=dataset,
            imputer=EchoImputer,
            classifier=ShapeCheckingClassifier,
            m=1,
            n_folds=2,
            target_level="variable",
        )

        test.run()

        expected_feature_width = len(dataset.get_predictor_cols_idx())
        expected_target_width = 1 + len(dataset.expl_vars)

        feature_counts = [
            x_shape[1] for x_shape, _ in ShapeCheckingClassifier.fit_records
        ]
        target_counts = [
            y_shape[1] if len(y_shape) == 2 else 1
            for _, y_shape in ShapeCheckingClassifier.fit_records
        ]

        self.assertEqual(set(feature_counts), {expected_feature_width})
        self.assertEqual(set(target_counts), {expected_target_width})
        self.assertEqual(len(feature_counts), test.n_folds * test.m * 2)

        predict_feature_counts = [shape[1] for shape in ShapeCheckingClassifier.predict_records]
        self.assertEqual(set(predict_feature_counts), {expected_feature_width})


if __name__ == "__main__":
    unittest.main()
