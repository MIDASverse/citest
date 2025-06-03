from citest.simulate import simulate
from citest.data import kuha
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np
import pandas as pd

import torch

np.random.seed(42)
torch.manual_seed(42)

ns = [200, 500, 1000, 2000, 5000]
R_bys = ["X", "Y"]
R_ins = ["X", "Y"]

B = 50

kuha_res = []
for n in ns:
    for rin in R_ins:
        for rby in R_bys:

            print(f"n={n}, R_in={rin}, R_by={rby}")

            res = simulate(
                kuha,
                n=n,
                B=B,
                imputer=IterativeImputer,
                classifier=RandomForest,
                dgp_args={"R_by": rby, "R_in": rin, "inc_Z": True},
                # imputer_args={"max_iter": 30},
                classifier_args={"n_estimators": 20, "n_jobs": 8},
            )

            res = pd.DataFrame(res)
            res["n"] = n
            res["R_in"] = rin
            res["R_by"] = rby

            print(res)
            kuha_res.append(res)

full_res = pd.concat(kuha_res, ignore_index=True)
full_res.to_csv("../../Results/kuha_june25.csv", index=False)
