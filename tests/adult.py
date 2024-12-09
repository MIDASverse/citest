import pandas as pd
import numpy as np

from citest.test import RLTest
from citest.imputer import IterativeImputer
from citest.classifier import RandomForest
from citest.data import v4_dgp, MAR1, Dataset


adult = pd.read_csv("citest/data/us-census-income.csv")
adult["income"] = adult["income"].map({"<=50K": 0, ">50K": 1})
adult = adult.iloc[:, :]
adult["income"] = np.where(adult["age"] > 50, np.nan, adult["age"])
adult.loc[0, "native-country"] = np.nan

for i in np.random.choice(adult.shape[0], adult.shape[0]):
    adult.iloc[i, np.random.choice(adult.shape[1], 1)] = np.nan

adult_data = Dataset()
adult_data.make(adult, y="income")

mod = RLTest(
    adult_data,
    imputer=IterativeImputer,
    classifier=RandomForest,
    n_folds=10,
    repetitions=10,
    classifier_args={"n_estimators": 20, "n_jobs": 8},
    imputer_args={"max_iter": 30},
)

mod.run()
mod.results
