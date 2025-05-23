import citest.test as test
import citest.mi_test as mi_test
from citest.data import v4_dgp, MAR1, adult, mushrooms
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np

B = 20
ps = [np.nan for _ in range(B)]

res = []
for b in range(B):
    print(b)
    # test_data = v4_dgp(5000, R_by="X", R_in="X")
    test_data = MAR1(5000, ci=True)
    # test1 = test.RLTest(
    test1 = mi_test.MITest(
        test_data,
        imputer=IterativeImputer,
        classifier=RandomForest,
        n_folds=10,
        m=10,
        classifier_args={"n_estimators": 20, "n_jobs": 8},
        imputer_args={"max_iter": 30},
        # imputer_args={"hidden_layers": [8, 32]},
    )

    test1.run()
    res.append(test1.results)
    print(test1.results["p"])

resdf = pd.DataFrame(res)

print(np.mean(resdf["p"] < 0.05))
