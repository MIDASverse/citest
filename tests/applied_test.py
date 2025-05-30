from citest.data import Dataset
from citest.imputer import MidasImputer
from citest.classifier import RandomForest
from citest.mi_test import MITest3

import pandas as pd
import numpy as np

pol_data = pd.read_csv("...")

pol_dataset = Dataset()
pol_dataset.make(pol_data, y="OUTCOME")

pol_test = MITest3(
    pol_dataset,
    imputer=MidasImputer,
    classifier=RandomForest,
    n_folds=10,
    m=10,
    n_jobs=8,
    verbose=True,
)

pol_test.run()
pol_test.summary()
