# Getting Started

## Installation

```bash
pip install citest
```

`citest` depends on PyTorch and MIDAS2 for the default imputer. These are installed automatically.

## Prepare your data

Start with a pandas DataFrame where missing values are encoded as `NaN`. Wrap it in a `Dataset` object:

```python
import pandas as pd
from citest.data import Dataset

data = pd.read_csv("my_data.csv")

dataset = Dataset()
dataset.make(
    data,
    y="target_variable",
    expl_vars=["x1", "x2", "x3"],
)
```

- **`y`** -- the outcome variable whose relationship to missingness you want to test.
- **`expl_vars`** -- covariates to condition on. If omitted, all columns except `y` are used.

Categorical columns are one-hot encoded automatically.

## Run the test

```python
from citest import CIMissTest

test = CIMissTest(
    dataset,
    classifier_args={"n_estimators": 20, "target_n_jobs": 8},
)
test.run()
```

This performs multiple imputation, trains classifiers to predict missingness with and without the outcome, and combines the results into a test statistic.

## Interpret the results

```python
test.summary()
```

The summary reports:

- **Mean difference in BCE** -- the average reduction in binary cross-entropy when the real outcome is included. Positive values indicate the outcome helps predict missingness.
- **t-statistic / p-value** -- a one-sided test of the null hypothesis that the outcome does not improve missingness prediction. A small *p*-value provides evidence against conditional independence (i.e. evidence of MNAR-type missingness).

## Key parameters

| Parameter | Default | Description |
|---|---|---|
| `m` | `10` | Number of multiply imputed datasets |
| `n_folds` | `10` | Number of cross-validation folds |
| `variance_method` | `"mi_crossfit"` | Variance estimator |
| `target_level` | `"variable"` | Missingness granularity: `"variable"` or `"column"` |
| `random_state` | `42` | Random seed for reproducibility |

## Choosing an imputer

| Class | Description | When to use |
|---|---|---|
| `MidasImputer` (default) | MIDAS denoising autoencoder via `MIDAS2` | General purpose; handles mixed types well |
| `IterativeImputer` | scikit-learn iterative imputer with posterior sampling | Faster; good for moderate-sized numeric data |
| `IterativeImputer2` | Robust variant with numerical guards | Wide or sparse data where `IterativeImputer` fails |

```python
from citest.imputer import IterativeImputer

test = CIMissTest(dataset, imputer=IterativeImputer)
```

## Choosing a classifier

| Class | Description | When to use |
|---|---|---|
| `RFClassifier` (default) | Random forest with auto-tuned hyperparameters | General purpose; robust default |
| `ETClassifier` | Extremely randomized trees | Faster training; more variance |
| `LogisticClassifier` | Logistic regression | Linear relationships; fast |

```python
from citest.classifier import RFClassifier

test = CIMissTest(
    dataset,
    classifier=RFClassifier,
    classifier_args={"n_estimators": 100, "target_n_jobs": 8},
)
```
