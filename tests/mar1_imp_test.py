from citest.data import MAR1
import statsmodels.api as sm
import numpy as np
import pandas as pd

B = 1000


def mar1_simulate(B=1000, ci=True):
    compl_coefs = [0] * B
    miss_coefs = [0] * B

    for i in range(1000):
        data = MAR1(1000, ci=ci)

        lin_compl = sm.OLS(
            data.full_data.iloc[:, 0],
            sm.add_constant(data.full_data.iloc[:, 1:], prepend=False),
        ).fit()
        lin_miss = sm.OLS(
            data.miss_data.iloc[:, 0],
            sm.add_constant(data.miss_data.iloc[:, 1:], prepend=False),
            missing="drop",
        ).fit()

        compl_coefs[i] = lin_compl.params
        miss_coefs[i] = lin_miss.params

    return compl_coefs, miss_coefs


compl_res_ci, miss_res_ci = (pd.DataFrame(x) for x in mar1_simulate(ci=True))
compl_res_nci, miss_res_nci = (pd.DataFrame(x) for x in mar1_simulate(ci=False))

compl_res_ci["ci"] = "CI"
compl_res_ci["missing"] = "Complete"
miss_res_ci["ci"] = "CI"
miss_res_ci["missing"] = "Missing"

compl_res_nci["ci"] = "NCI"
compl_res_nci["missing"] = "Complete"
miss_res_nci["ci"] = "NCI"
miss_res_nci["missing"] = "Missing"

results = pd.concat([compl_res_ci, miss_res_ci, compl_res_nci, miss_res_nci])
results.to_csv("../../Results/mar1_reg_comparison.csv")
