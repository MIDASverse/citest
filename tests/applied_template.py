from citest.data import Dataset
from citest.imputer import MidasImputer
from citest.classifier import RandomForest
from citest.mi_test import MITest2

import pandas as pd
import numpy as np
import random

# Set seed
import torch

np.random.seed(42)
torch.manual_seed(42)

# Load in the dataset using pandas
pol_data = pd.read_csv(
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/AJPS/Aklin and Kern 2021/ak2021_mod3.csv"
)

## NOTE: Make sure the dataset columns have the correct types (int/float, boolean, category)

# Define the dataset object
pol_dataset = Dataset()
# TODO: Allow for restricting variables for CI aspect only
pol_dataset.make(
    pol_data, y="finreform", expl_vars=["var1", "var2"]
)  # You must specify the outcome variable


# Define the test object
pol_test = MITest2(  # TODO: Update name of test function
    pol_dataset,
    imputer=MidasImputer,
    classifier=RandomForest,
    n_folds=10,
    m=10,
    # verbose=True,
    classifier_args={"n_estimators": 20, "n_jobs": 8},
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results
