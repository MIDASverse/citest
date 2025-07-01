import unittest
from citest.data import *
import pandas as pd


class TestDataset(unittest.TestCase):

    def setUp(self):
        """Create a fake dataset for testing."""
        self.fake_data = pd.DataFrame(
            {
                "A": [1, 2, 3, np.nan, 5],
                "B": [np.nan, 2, np.nan, 4, 5],
                "C": ["x", np.nan, "y", "z", np.nan],
                "D": [1.0, 2.0, 3.0, 4.0, 5.0],
                "Y": [1, 0, 1, 0, 1],
            }
        )

        self.dataset = Dataset()
        self.dataset.make(data=self.fake_data, y="Y", expl_vars=["A", "C"])

    def test_make(self):
        """Test the make method of the Dataset class."""

        self.assertIsInstance(self.dataset, Dataset)

    def test_make_arrangement(self):

        self.assertEqual(self.dataset.miss_data.columns[0], "Y")
        self.assertEqual(self.dataset.miss_data.shape, (5, 7))

    def test_make_expl_vars(self):

        self.assertEqual(self.dataset.expl_vars, ["A", "C"])
        self.assertTrue((self.dataset._expl_vars == [1, 4, 5, 6]).all())

    def test_make_float_preservation(self):
        self.assertIsInstance(self.dataset.miss_data.A, pd.Series)
        self.assertTrue(pd.api.types.is_float_dtype(self.dataset.miss_data.A))

    def test_make_missingness(self):
        self.assertEqual(self.dataset.mask.sum(), (35 - 9))

    def test_y_failcases(self):
        """Test the make method of the Dataset class."""
        fake_data = pd.DataFrame(
            {
                "A": [1, 2, 3, np.nan, 5],
                "Y": [1, 0, 1, 0, 1],
            }
        )

        citest_dataset = Dataset()
        with self.assertRaises(TypeError):
            citest_dataset.make(data=fake_data, expl_vars=["A", "C"])
        with self.assertRaises(ValueError):
            citest_dataset.make(data=fake_data, y="C", expl_vars=["A", "C"])

    def test_make_no_expl_vars(self):
        """Test the make method of the Dataset class with no expl_vars."""
        citest_dataset = Dataset()
        citest_dataset.make(data=self.fake_data, y="Y")

        self.assertEqual(citest_dataset.expl_vars, ["A", "B", "C", "D"])
        self.assertTrue((citest_dataset._expl_vars == [1, 2, 4, 5, 6, 3]).all())


if __name__ == "__main__":
    unittest.main()
