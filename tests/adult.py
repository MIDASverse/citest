import numpy as np

from citest.imputer import IterativeImputer, CompleteImputer, MidasImputer
from citest.classifier import RandomForest
from citest.data import adult

from citest.simulate import simulate

np.random.seed(89)

adult_res = simulate(
    adult,
    n=1000,
    B=3,
    imputer=MidasImputer,
    classifier=RandomForest,
    dgp_args={"mcar_prop": 0.5, "ci": True},
    # imputer_args={"max_iter": 30},
    classifier_args={"n_estimators": 5, "n_jobs": 8},
)

np.mean([p < 0.05 for p in adult_res])
