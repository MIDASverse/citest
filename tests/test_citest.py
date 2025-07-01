import unittest
import numpy as np
import pandas as pd
from citest.mi_test import MITest2
from citest.data import Dataset
from citest.imputer import IterativeImputer


class TestMITest2(unittest.TestCase):
    def setUp(self):
        # Create a mock dataset
        data = pd.DataFrame(np.random.rand(100, 5))
        mask = np.random.choice([0, 1], size=data.shape, p=[0.2, 0.8])
        data[mask == 0] = np.nan
        data.columns = ["y", "x1", "x2", "x3", "x4"]
        self.dataset = Dataset()
        self.dataset.make(data, y="y", expl_vars=["x1", "x3", "x4"])

        # Initialize MITest2 with the mock dataset
        self.test = MITest2(
            dataset=self.dataset,
            imputer=IterativeImputer,
            n_folds=2,
            m=2,
            imputer_args={"max_iter": 2},
            classifier_args={"n_estimators": 3},
        )

        self.result_return = self.test.run()

    def test_run_test(self):
        # Run the test method and check if it completes without errors

        self.assertIsNone(self.result_return)

    def test_results(self):
        # Run the test and then check the summary
        self.assertIsInstance(self.test.results, dict)
        self.assertIsInstance(self.test.results["m"], np.float64)

    def test_expl_vars(self):
        # Check if the explanatory variables are correctly set
        self.assertEqual(self.test.dataset.expl_vars, ["x1", "x3", "x4"])
        self.assertTrue(
            all(
                var in self.test.dataset.miss_data.columns
                for var in self.test.dataset.expl_vars
            )
        )


if __name__ == "__main__":
    unittest.main()
