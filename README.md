# citest: Conditional Independence Testing for Missing Data

*This repo is currently private.*

Our test provides a statistical estimate of whether an outcome is independent of missing values, conditional on the observed explanatory data. 

## Testing

To use the test, download this repo. From the main directory, you can import the citest module:

```python
from citest.test import RLtest
from citest.data import v4_dgp, MAR1
from citest.classifier import RandomForest
from citest.imputer import *

import numpy as np

test_data = MAR1(100, ci = False)
nci_ex = RLTest(
    test_data,
    imputer=IterativeImputer,
    classifier=RandomForest,
    n_folds=10,
    repetitions=10,
    classifier_args={"n_estimators": 20},
    imputer_args={"max_iter": 30},
)

nci_ex.run()
nci_ex.summary()

```