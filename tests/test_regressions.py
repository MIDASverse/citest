import unittest

import numpy as np
import pandas as pd

from citest.classifier import RFClassifier
from citest.data import Dataset
from citest.imputer import NullImputer, IterativeImputer
from citest.test import CIMissTest
from citest.utils import BCEclip


def make_small_dataset():
    data = pd.DataFrame(
        {
            "Y": [1, 0, 1, 0],
            "A": [0.5, np.nan, 0.1, 1.0],
            "B": [1, 2, 3, np.nan],
        }
    )
    ds = Dataset()
    ds.make(data, y="Y", expl_vars=["A", "B"], _onehot=False)
    return ds


class RegressionTests(unittest.TestCase):
    def test_random_forest_predict_shapes_single_and_multi(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(10, 3))

        y_single = rng.integers(0, 2, size=10)
        clf_single = RFClassifier(n_estimators=5, random_state=0)
        clf_single.fit(X, y_single)
        preds_single = clf_single.predict(X)
        self.assertEqual(preds_single.shape, (10, 1))

        y_multi = np.column_stack(
            [rng.integers(0, 2, size=10), rng.integers(0, 2, size=10)]
        )
        clf_multi = RFClassifier(n_estimators=5, random_state=1)
        clf_multi.fit(X, y_multi)
        preds_multi = clf_multi.predict(X)
        self.assertEqual(preds_multi.shape, y_multi.shape)

    def test_get_m_complete_returns_m_independent_frames(self):
        ds = make_small_dataset()

        null_imputer = NullImputer(dataset=ds)
        null_imps = null_imputer.get_m_complete(m=3)
        self.assertEqual(len(null_imps), 3)
        self.assertTrue(all(frame.shape == ds.miss_data.shape for frame in null_imps))

        iter_imputer = IterativeImputer(dataset=ds)
        iter_imps = list(iter_imputer.get_m_complete(m=2, max_iter=1))
        self.assertEqual(len(iter_imps), 2)
        self.assertTrue(all(frame.shape == ds.miss_data.shape for frame in iter_imps))
        self.assertIsNot(iter_imps[0], iter_imps[1])

    def test_bceclip_handles_extremes(self):
        p = np.array([[0.0, 1.0], [1.0, 0.0]])
        y = np.array([[0, 1], [1, 0]])
        clipped = BCEclip(p, y)
        self.assertEqual(clipped.shape, p.shape)
        self.assertTrue(np.isfinite(clipped).all())

    def test_cimiss_small_sample_fold_guard(self):
        ds = make_small_dataset()
        with self.assertRaises(ValueError):
            CIMissTest(
                dataset=ds, imputer=NullImputer, classifier=RFClassifier, n_folds=10
            ).run()

        # n_folds equal to n should be permitted
        CIMissTest(
            dataset=ds, imputer=NullImputer, classifier=RFClassifier, n_folds=4, m=1
        ).run()

    def test_cimiss_minimal_integration(self):
        ds = make_small_dataset()
        test = CIMissTest(
            dataset=ds,
            imputer=NullImputer,
            classifier=RFClassifier,
            m=1,
            n_folds=2,
            classifier_args={"n_estimators": 5, "random_state": 0},
        )
        test.run()
        self.assertIsInstance(test.results, dict)
        for key in ["m", "B", "W_bar", "T", "t_k", "p_k", "p_2s"]:
            self.assertIn(key, test.results)


if __name__ == "__main__":
    unittest.main()
