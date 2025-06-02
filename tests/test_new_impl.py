import citest.test as test
import citest.mi_test as mi_test
from citest.data import v4_dgp, MAR1, adult, mushrooms, kuha
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np
import pandas as pd

B = 20
ps = [np.nan for _ in range(B)]

res = []

for b in range(B):
    print(b)
    # test_data = kuha(2000, R_by="X", R_in="X", inc_Z=True)
    test_data = adult(2000, ci=True)
    test_obj = mi_test.MITest2(
        test_data,
        imputer=MidasImputer,
        classifier=RandomForest,
        n_folds=10,
        m=10,
        classifier_args={"n_estimators": 20, "n_jobs": 8},
        # imputer_args={"max_iter": 30},
        # imputer_args={"device": 'cpu'},
    )
    test_obj.run()
    res.append(test_obj.results)

res_df = pd.DataFrame(res)

(res_df.p_m < 0.05).mean()
(res_df.p_k < 0.05).mean()
