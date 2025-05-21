import citest.test as test
from citest.data import v4_dgp, MAR1, adult
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np

B = 5
ps = [np.nan for _ in range(B)]

for b in range(B):
    print(b)
    # test_data = v4_dgp(2000, R_by="X", R_in="X")
    test_data = adult(2000, ci=False)
    test1 = test.RLTest(
        test_data,
        imputer=IterativeImputer,
        classifier=RandomForest,
        n_folds=10,
        repetitions=10,
        classifier_args={"n_estimators": 20, "n_jobs": 8},
        imputer_args={"max_iter": 100},
        # imputer_args={"hidden_layers": [8, 4]},
    )

    test1.run()
    ps[b] = test1.results["p"]

print(np.mean(np.array(ps) < 0.05))
