import citest.test as test
from citest.data import v4_dgp, MAR1
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np


def simulate(
    DGP,
    n,
    dgp_args={},
    imputer=IterativeImputer,
    imputer_args={},
    classifier=RandomForest,
    classifier_args={},
    B=200,
):
    ps = [np.nan for _ in range(B)]

    for b in range(B):
        print(b)
        test_data = DGP(n, **dgp_args)

        test1 = test.RLTest(
            test_data,
            imputer=imputer,
            classifier=classifier,
            n_folds=10,
            repetitions=10,
            classifier_args={**classifier_args},
            imputer_args={**imputer_args},
        )

        test1.run()
        ps[b] = test1.results["p"]

    return ps
