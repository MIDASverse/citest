import io
import unittest
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

from citest.data import Dataset
from citest.imputer import (
    Imputer,
    CompleteImputer,
    NullImputer,
    IterativeImputer,
)
from citest.classifier import CIClassifier, RandomForest
from citest.test import CIMissTest


class DummyImputer(Imputer):
    """Minimal imputer that just fills NaNs with -1 for testing."""

    def _complete(self, **kwargs):
        # Mark completion by filling missing entries with -1
        self.completed = self.dataset.miss_data.fillna(-1)

    def get_m_complete(self, m: int = 1, train_index=None, **kwargs):
        # Always return copies of the completed frame so shapes remain consistent
        if self.completed is None:
            self._complete(**kwargs)
        return [self.completed.copy() for _ in range(m)]


class DummyClassifier(CIClassifier):
    """Classifier stub that returns constant probabilities."""

    def __init__(self, random_state=None, **kwargs):
        super().__init__()
        self.target_width = None
        self.was_fit = False
        self.random_state = None

    def _fit(self, X, y):
        # Record the target width so predictions can match the training mask
        self.target_width = y.shape[1] if y.ndim > 1 else 1
        self.was_fit = True

    def _predict(self, X):
        # Emit a matrix of 0.5 probabilities with the expected shape
        return np.full((X.shape[0], self.target_width), 0.5)


def make_tiny_dataset():
    """Helper to construct a small dataset with a mix of numeric and categorical data."""

    data = pd.DataFrame(
        {
            "Y": [1, 0, 1, 0],
            "X1": [1.0, np.nan, 3.0, 4.0],
            "X2": ["a", "b", "a", "b"],
        }
    )
    ds = Dataset()
    ds.make(data, y="Y")
    return ds


class DatasetTests(unittest.TestCase):
    def test_make_sets_mask_and_shape(self):
        # The mask should flag non-null entries and have the same shape as the data
        ds = make_tiny_dataset()
        self.assertEqual(ds.miss_data.shape, (4, 4))
        self.assertEqual(ds.mask.shape, ds.miss_data.shape)
        self.assertTrue(ds.mask[0, 0])

    def test_make_blocks_double_initialization(self):
        # Calling make twice on the same instance should raise to prevent accidental reuse
        ds = make_tiny_dataset()
        with self.assertRaises(ValueError):
            ds.make(ds.miss_data, y="Y")

    def test_dummy_one_hot_tracks_expl_vars(self):
        # The one-hot encoding should expand categoricals and track explanatory indices
        ds = make_tiny_dataset()
        self.assertGreater(len(ds._expl_vars), 0)
        self.assertTrue(
            all(isinstance(idx, (int, np.integer)) for idx in ds._expl_vars)
        )


class ImputerTests(unittest.TestCase):
    def test_complete_imputer_uses_full_data(self):
        # CompleteImputer should immediately expose the provided full_data
        ds = make_tiny_dataset()
        ds.full_data = ds.miss_data.copy()
        imp = CompleteImputer(dataset=ds)
        completed = imp.get_m_complete(m=1)[0]
        pd.testing.assert_frame_equal(completed, ds.full_data)

    def test_null_imputer_replaces_missing_with_zero(self):
        # NullImputer should replace only the missing entries with zeros
        ds = make_tiny_dataset()
        imp = NullImputer(dataset=ds)
        completed = imp.completed
        self.assertFalse(completed.isna().any().any())
        self.assertTrue(
            (completed.loc[1, "X1"] == 0) and (completed.loc[0, "X1"] == 1.0)
        )

    def test_iterative_imputer_sets_model(self):
        # The iterative imputer should fit a model and produce completed draws
        ds = make_tiny_dataset()
        imp = IterativeImputer(dataset=ds)
        imps = imp.get_m_complete(m=2, max_iter=1)
        self.assertIsNotNone(imp.model)
        self.assertEqual(len(imps), 2)
        self.assertTrue(all(frame.shape == ds.miss_data.shape for frame in imps))


class ClassifierTests(unittest.TestCase):
    def test_random_forest_predicts_probability_matrix(self):
        # Predict should return probabilities with one column per target in multi-output data
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 3))
        y = np.column_stack([rng.integers(0, 2, size=20), rng.integers(0, 2, size=20)])
        clf = RandomForest(n_estimators=5, random_state=0)
        clf.fit(X, y)
        preds = clf.predict(X)
        self.assertEqual(preds.shape, y.shape)
        self.assertTrue(np.logical_and(preds >= 0, preds <= 1).all())


class CIMissTestTests(unittest.TestCase):
    def test_run_produces_results(self):
        # Running the test should populate the results dictionary with expected keys
        ds = make_tiny_dataset()
        ci_test = CIMissTest(
            dataset=ds,
            imputer=DummyImputer,
            classifier=DummyClassifier,
            m=1,
            n_folds=2,
        )
        ci_test.run()
        self.assertIsInstance(ci_test.results, dict)
        for key in ["m", "B", "W_bar", "T", "t_k", "p_k", "p_2s"]:
            self.assertIn(key, ci_test.results)

    def test_summary_runs_after_test(self):
        # summary should not error once run has been called
        ds = make_tiny_dataset()
        ci_test = CIMissTest(
            dataset=ds,
            imputer=DummyImputer,
            classifier=DummyClassifier,
            m=1,
            n_folds=2,
        )
        ci_test.run()
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            ci_test.summary()
        self.assertTrue(len(buffer.getvalue()) > 0)


if __name__ == "__main__":
    unittest.main()
