# citest: Conditional Independence Testing for Missing Data

*This repo is currently private.*

Our test provides a statistical estimate of whether an outcome is independent of missing values, conditional on the observed explanatory data. 

## Testing

To use the test, download this repo. From the main directory, you can import the citest module (all names are temporary for the time being):

```python
from citest.mi_test import MITest2
from citest.data import MAR1
from citest.classifier import RandomForest
from citest.imputer import IterativeImputer

import numpy as np

# generate some test data from King (2001)
test_data = MAR1(100, ci = False)

# declare the test object
nci_ex = MITest2(
    test_data,
    imputer=IterativeImputer,
    classifier=RandomForest,
    n_folds=10,
    m=10,
    classifier_args={"n_estimators": 20},
    imputer_args={"max_iter": 30},
)

# run the test
nci_ex.run()

# get the results
nci_ex.summary()

```