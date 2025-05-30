import citest.test as test
import citest.mi_test as mi_test
from citest.data import v4_dgp, MAR1, adult, mushrooms
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np
import pandas as pd

B = 3
ps = [np.nan for _ in range(B)]

res2 = []
res3 = []
for b in range(B):
    print(b)
    # test_data = v4_dgp(500, R_by="X", R_in="Y")
    test_data = mushrooms(2000, ci=True)
    # test1 = test.RLTest(
    test_mi2 = mi_test.MITest2(
        test_data,
        imputer=MidasImputer,
        classifier=RandomForest,
        n_folds=10,
        m=10,
        classifier_args={"n_estimators": 20, "n_jobs": 8},
        # imputer_args={"max_iter": 30},
        # imputer_args={},
    )
    test_mi2.run()
    res2.append(test_mi2.results)

    test_mi3 = mi_test.MITest3(
        test_data,
        imputer=MidasImputer,
        classifier=RandomForest,
        n_folds=10,
        m=10,
        classifier_args={"n_estimators": 20, "n_jobs": 8},
        # imputer_args={"max_iter": 30},
        # imputer_args={},
    )
    test_mi3.run()
    res3.append(test_mi3.results)

test2_df = pd.DataFrame(res2)
test3_df = pd.DataFrame(res3)

(test2_df.p < 0.05).mean()
(test3_df.p < 0.05).mean()
