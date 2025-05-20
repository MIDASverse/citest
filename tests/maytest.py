from citest.imputer import IterativeImputer, MidasImputer
from citest.classifier import RandomForest
from citest.data import MAR1, v4_dgp, adult
from citest.test_subsample import RLTest

import pandas as pd
import numpy as np
from scipy import stats

if __name__ == "__main__":

    B = 2

    type_dfs = []
    for type in ["X", "Y"]:
        # for type in [True, False]:
        res = [np.NaN] * B

        for b in range(B):
            print(b)
            test_data = v4_dgp(2000, R_by=type, R_in="X")
            # test_data = MAR1(2000, ci=type)
            test1 = RLTest(
                test_data,
                imputer=IterativeImputer,
                classifier=RandomForest,
                n_folds=10,
                repetitions=10,
                imputer_args={"max_iter": 30},
                # imputer_args={"hidden_layers": [256, 128]},
                classifier_args={"n_estimators": 20, "n_jobs": 8},
            )

            test1.run()
            res[b] = test1.results

        resdf = pd.DataFrame(res)
        resdf["type"] = "null" if type == "X" else "alt"
        # resdf["type"] = "null" if type else "alt"
        type_dfs.append(resdf)

    resall = pd.concat(type_dfs, axis=0)

    resall.groupby("type").agg(
        p_prop=("p", lambda x: (x < 0.05).mean()),
    )
