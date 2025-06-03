from citest.simulate import simulate
from citest.data import MAR1
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np
import pandas as pd

import torch

np.random.seed(42)
torch.manual_seed(42)

ns = [200, 500, 1000, 2000, 5000]
cis = [True, False]

B = 50

MAR1_res = []
for n in ns:
    for ci in cis:

        print(f"n={n}, ci={ci}")

        res = simulate(
            MAR1,
            n=n,
            B=B,
            imputer=MidasImputer,
            classifier=RandomForest,
            dgp_args={"ci": ci},
            imputer_args={},
            classifier_args={"n_estimators": 20, "n_jobs": 8},
        )

        res = pd.DataFrame(res)
        res["n"] = n
        res["ci"] = ci

        MAR1_res.append(res)

full_res = pd.concat(MAR1_res, ignore_index=True)
full_res.to_csv("../../Results/MAR1_june25.csv", index=False)
