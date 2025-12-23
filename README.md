# citest: Conditional Independence Testing for Missing Data

*This repo is currently private.*

Our test provides a statistical estimate of whether an outcome is independent of missing values, conditional on the observed explanatory data. 

## Dummy example

To use the test, download this repo. From the main directory, you can import the citest module (all names are temporary for the time being):

```python
import pandas as pd
from citest import CIMissTest
from citest.data import Dataset

# Load in dataset using pandas
appl_data = pd.read_csv("path/to/your/data.csv")

# Define the dataset object
appl_dataset = Dataset()
appl_dataset.make(
    appl_data, 
    y="target_variable", 
    expl_vars=["expl_var1", "expl_var2", ...]
)

# Define the test object
appl_test = CIMissTest(
    appl_dataset,
    classifier_args = {"n_estimators": 20, "n_jobs": 8},
)

# Run the test
appl_test.run()

# Print a summary of the results
appl_test.summary()

```

### Customizing the imputation and classification models

We have tuned the defaults to work in sensible, applied settings. However, users can customize the imputation and classification models by passing keyword arguments to the `CIMissTest` object:

```python
from citest.imputer import MidasImputer
from citest.classifier import RandomForest

appl_test2 = CIMissTest(
    appl_dataset,
    imputer = MIDASImputer,
    classifier = RandomForest,
    n_folds=10,
    m=10,
    classifier_args = {"n_estimators": 20, "n_jobs": 8},
    imputer_args = {"hidden_layers": [8,4,2], "epochs": 500},
)
```
