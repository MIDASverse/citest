import unittest

import numpy as np

from citest import data
from citest.data import Dataset


class TestDGPSmoke(unittest.TestCase):
    def _assert_basic_dataset(self, ds: Dataset, expected_n: int):
        self.assertIsInstance(ds, Dataset)
        self.assertIsNotNone(ds.miss_data)
        self.assertEqual(ds.miss_data.shape[0], expected_n)
        self.assertEqual(ds.mask.shape, ds.miss_data.shape)
        self.assertEqual(ds.miss_data.columns[0], ds.y_name)
        if ds.full_data is not None:
            self.assertEqual(ds.full_data.shape, ds.miss_data.shape)
        if ds.weights is not None:
            self.assertEqual(ds.weights.shape[0], ds.miss_data.shape[1])

    def test_identify(self):
        np.random.seed(2)
        for ci in [True, False]:
            with self.subTest(ci=ci):
                ds = data.identify(n=30, ci=ci, eta=0.1)
                self._assert_basic_dataset(ds, 30)

    def test_single_mar(self):
        np.random.seed(3)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.single_mar(n=30, ci=ci, missing_mech=mech)
                    self._assert_basic_dataset(ds, 30)

    def test_single_mnar(self):
        np.random.seed(4)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.single_mnar(n=30, ci=ci, missing_mech=mech)
                    self._assert_basic_dataset(ds, 30)

    def test_mar1_mnar1(self):
        np.random.seed(5)
        for fn in [data.MAR1, data.MNAR1]:
            for ci in [True, False]:
                for mech in ["linear", "xor"]:
                    with self.subTest(fn=fn.__name__, ci=ci, mech=mech):
                        ds = fn(n=30, ci=ci, missing_mech=mech)
                        self._assert_basic_dataset(ds, 30)

    def test_adult(self):
        np.random.seed(6)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.adult(n=50, ci=ci, mcar_prop=0.1, missing_mech=mech)
                    self._assert_basic_dataset(ds, 50)

    def test_adult_mnar(self):
        np.random.seed(7)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.adult_mnar(
                        n=50, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 50)

    def test_mushrooms(self):
        np.random.seed(8)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.mushrooms(
                        n=50, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 50)

    def test_breast_cancer(self):
        np.random.seed(9)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.breast_cancer(
                        n=50, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 50)

    def test_wine(self):
        np.random.seed(10)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.wine(n=50, ci=ci, mcar_prop=0.1, missing_mech=mech)
                    self._assert_basic_dataset(ds, 50)

    def test_diabetes(self):
        np.random.seed(11)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.diabetes(n=50, ci=ci, mcar_prop=0.1, missing_mech=mech)
                    self._assert_basic_dataset(ds, 50)

    def test_covertype(self):
        np.random.seed(12)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.covertype(n=100, ci=ci, mcar_prop=0.1, missing_mech=mech)
                    self._assert_basic_dataset(ds, 100)

    def test_california_housing(self):
        np.random.seed(13)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.california_housing(
                        n=100, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 100)

    def test_german_credit(self):
        np.random.seed(14)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.german_credit(
                        n=200, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 200)

    def test_bank_marketing(self):
        np.random.seed(15)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.bank_marketing(
                        n=200, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 200)

    def test_ames_housing(self):
        np.random.seed(16)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.ames_housing(
                        n=200, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 200)

    def test_give_me_some_credit(self):
        np.random.seed(17)
        for ci in [True, False]:
            for mech in ["linear", "xor"]:
                with self.subTest(ci=ci, mech=mech):
                    ds = data.give_me_some_credit(
                        n=200, ci=ci, mcar_prop=0.1, missing_mech=mech
                    )
                    self._assert_basic_dataset(ds, 200)


if __name__ == "__main__":
    unittest.main()
