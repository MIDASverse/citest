from citest.simulate import simulate
from citest.imputer import IterativeImputer
from citest.classifier import RandomForest
from citest.data import v4_dgp, MAR1

import pandas as pd
import numpy as np

if __name__ == "__main__":

    ns = [100, 200, 500, 1000, 2000, 5000, 10000]
    R_bys = ["X", "Y"]
    R_ins = ["X", "Y"]

    np.random.seed(89)

    v4_res = {}
    for n in ns:
        for rin in R_ins:
            for rby in R_bys:
                v4_res[str(n) + "_by" + rby + "_in" + rin] = simulate(
                    v4_dgp,
                    n,
                    B=200,
                    imputer=IterativeImputer,
                    classifier=RandomForest,
                    dgp_args={"R_by": rby, "R_in": rin},
                    imputer_args={"max_iter": 30},
                    classifier_args={"n_estimators": 20, "n_jobs": 8},
                )

    pd.DataFrame(v4_res).to_csv("../../Results/v4_dgp_results_by_n.csv")

    mar1_res = {}
    for n in ns:
        for ci in [True, False]:
            mar1_res[str(n) + "_" + ("ci" if ci else "nci")] = simulate(
                MAR1,
                n,
                B=200,
                imputer=IterativeImputer,
                classifier=RandomForest,
                dgp_args={"ci": ci},
                imputer_args={"max_iter": 30},
                classifier_args={"n_estimators": 20, "n_jobs": 8},
            )

    pd.DataFrame(mar1_res).to_csv("../../Results/mar1_dgp_results_by_n.csv")

    mar1_nest_res = {}
    n_estimators = [5, 10, 20, 50, 100]
    for n_est in n_estimators:
        for ci in [True, False]:
            mar1_nest_res[str(n_est) + "_" + ("ci" if ci else "nci")] = simulate(
                MAR1,
                n=2000,
                B=200,
                imputer=IterativeImputer,
                classifier=RandomForest,
                dgp_args={"ci": ci},
                imputer_args={"max_iter": 30},
                classifier_args={"n_estimators": n_est, "n_jobs": 8},
            )

    pd.DataFrame(mar1_nest_res).to_csv(
        "../../Results/mar1_dgp_results_by_nestimators.csv"
    )
