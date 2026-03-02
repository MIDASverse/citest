# citest: Conditional Independence Testing for Missing Data

A hypothesis test for whether an outcome variable is independent of missingness, conditional on the observed explanatory data. The test compares classifier performance in predicting missingness with and without the outcome variable, using multiple imputation and cross-fitting to produce a valid *t*-statistic and *p*-value.

## Installation

```bash
pip install citest
```

## Quick start

```python
import pandas as pd
from citest import CIMissTest
from citest.data import Dataset

# Load your data
data = pd.read_csv("path/to/your/data.csv")

# Define the dataset
dataset = Dataset()
dataset.make(
    data,
    y="target_variable",
    expl_vars=["expl_var1", "expl_var2", ...]
)

# Run the test
test = CIMissTest(
    dataset,
    classifier_args={"n_estimators": 20, "target_n_jobs": 8},
)
test.run()

# Print results
test.summary()
```

## How the test works

1. **Multiple imputation** -- the missing data are multiply imputed (default: MIDAS denoising autoencoder).
2. **Classifier comparison** -- for each imputed dataset, two classifiers predict the missingness indicator *R*:
   - One using the outcome *Y* and covariates *X*
   - One using a permuted (uninformative) copy of *Y* and covariates *X*
3. **Cross-fitting** -- predictions are made out-of-fold to avoid data leakage.
4. **Test statistic** -- the weighted difference in binary cross-entropy between the two classifiers is combined across imputations using Rubin's rules, yielding a *t*-statistic and *p*-value.

A significant result indicates that missingness depends on the outcome even after conditioning on the covariates (i.e. the data are not missing at random with respect to *Y*).

## Customizing the pipeline

### Imputers

| Class | Description |
|---|---|
| `MidasImputer` (default) | MIDAS denoising autoencoder (via `MIDAS2`) |
| `IterativeImputer` | scikit-learn iterative imputer with posterior sampling |
| `IterativeImputer2` | Robust variant with numerical guards for wide/sparse data |

### Classifiers

| Class | Description |
|---|---|
| `RFClassifier` (default) | Random forest with auto-tuned `max_features` and `min_samples_leaf` |
| `ETClassifier` | Extremely randomized trees |
| `LogisticClassifier` | Logistic regression |

### Example with custom settings

```python
from citest.imputer import IterativeImputer
from citest.classifier import RFClassifier

test = CIMissTest(
    dataset,
    imputer=IterativeImputer,
    classifier=RFClassifier,
    n_folds=10,
    m=10,
    classifier_args={"n_estimators": 100, "target_n_jobs": 8},
    imputer_args={"max_iter": 20},
)
```

### Key parameters

| Parameter | Default | Description |
|---|---|---|
| `m` | 10 | Number of multiply imputed datasets |
| `n_folds` | 10 | Number of cross-validation folds |
| `variance_method` | `"mi_crossfit"` | Variance estimator |
| `target_level` | `"variable"` | Granularity of the missingness target: `"variable"` or `"column"` |
| `random_state` | 42 | Random seed for reproducibility |

## Interpreting results

`test.summary()` prints the test output:

- **Mean difference in BCE** -- average reduction in cross-entropy when the real outcome is included. Positive values indicate the outcome helps predict missingness.
- **t / p-value** -- one-sided test of H0: the outcome does not improve missingness prediction. A small *p*-value provides evidence against conditional independence (i.e. evidence of MNAR-type missingness).

## License

MIT
