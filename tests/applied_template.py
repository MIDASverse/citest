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
    "/Users/ranjitlall/Library/CloudStorage/Dropbox/Ranjit's work/Harvard/Missing data/Missingness Tests/Applied data/AJPS/Clark and Dolan 2021/cd2021_full.csv"
)

## NOTE: Make sure the dataset columns have the correct types (int/float, boolean, category)

# Define the dataset object
pol_dataset = Dataset()
pol_dataset.make(
    pol_data, y="count_pa", expl_vars=["absidealimportantdiff", "board", "colony", "unsc", "USaid", "CHaid", "gdppc", "dservtoGDP", "dshorttoexports", "inflation", "debttoGDP", "FDItoGDP", "polity2", "openness", "war", "elec", "IMF", "crisis", "ccode"]
)  # You must specify the outcome variable

# Define the test object
pol_test = MITest2(
    pol_dataset,
#    imputer=MidasImputer,
#    classifier=RandomForest,
#    n_folds=10,
#    m=10,
#    classifier_args={"n_estimators": 20, "n_jobs": 8},
#    imputer_args={"hidden_layers": [8, 4, 2], "epochs": 500},
)

pol_test.run()  # Will run k * m tests
pol_test.summary()  # Will print a nice summary of results
