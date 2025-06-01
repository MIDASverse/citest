import citest.test as test
import citest.mi_test as mi_test
from citest.data import v4_dgp, MAR1, adult, mushrooms, kuha
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np
import pandas as pd

B = 5
ps = [np.nan for _ in range(B)]

res = []

for b in range(B):
    print(b)
    test_data = kuha(2000, R_by="X", R_in="Y", inc_Z=True)
    # test_data = mushrooms(500, ci=True)
    # test_obj = test.RLTest(
    test_obj = mi_test.MITest2(
        test_data,
        imputer=IterativeImputer,
        classifier=RandomForest,
        n_folds=10,
        m=10,
        classifier_args={"n_estimators": 20, "n_jobs": 8},
        # imputer_args={"max_iter": 30},
        # imputer_args={},
    )
    test_obj.run()
    res.append(test_obj.results)

res_df = pd.DataFrame(res)

(res_df.p < 0.05).mean()
