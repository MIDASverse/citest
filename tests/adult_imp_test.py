from citest.data import adult
import statsmodels.api as sm
import numpy as np
import pandas as pd

B = 200

coef_diff = np.zeros(B)

for b in range(B):
    data = adult(1000, mcar_prop=0.01)

    oh_data = data._dummy(data.full_data, drop_first=True)
    oh_data = pd.concat([oh_data["income"], oh_data.drop("income", axis=1)], axis=1)

    oh_y = oh_data["income"]
    oh_X = sm.add_constant(oh_data.drop("income", axis=1).astype(float), prepend=False)

    lin_compl = sm.OLS(
        oh_y,
        oh_X,
    ).fit()

    missing_rows = data.miss_data.isnull().any(axis=1)
    lin_miss = sm.OLS(
        oh_y[~missing_rows.values],
        oh_X[~missing_rows.values],
    ).fit()

    coef_diff[b] = lin_compl.params[0] - lin_miss.params[0]

coef_diff.mean()
