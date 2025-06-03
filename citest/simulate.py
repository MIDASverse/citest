from citest.mi_test import MITest2
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
    res = [np.nan for _ in range(B)]

    for b in range(B):
        print(b)
        test_data = DGP(n, **dgp_args)

        test1 = MITest2(
            test_data,
            imputer=imputer,
            classifier=classifier,
            n_folds=10,
            m=10,
            classifier_args={**classifier_args},
            imputer_args={**imputer_args},
        )

        test1.run()
        res[b] = test1.results

    return res
