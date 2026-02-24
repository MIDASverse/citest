# Examples

## 1. Simulated data -- null is true (MAR)

When conditional independence holds, the test should produce a non-significant *p*-value.

```python
from citest import CIMissTest
from citest.data import MAR1

# ci=True means conditional independence holds
dataset = MAR1(n=1000, ci=True)

test = CIMissTest(
    dataset,
    m=10,
    n_folds=10,
    classifier_args={"n_estimators": 20, "n_jobs": 8},
)
test.run()
test.summary()
```

Expected output: a large *p*-value (e.g. > 0.05), indicating no evidence against conditional independence.

## 2. Simulated data -- alternative is true (MNAR)

When the outcome influences missingness, the test should reject.

```python
from citest.data import MAR1

# ci=False means missingness depends on Y
dataset = MAR1(n=1000, ci=False)

test = CIMissTest(
    dataset,
    m=10,
    n_folds=10,
    classifier_args={"n_estimators": 20, "n_jobs": 8},
)
test.run()
test.summary()
```

Expected output: a small *p*-value (e.g. < 0.05), indicating evidence against conditional independence.

## 3. Real data -- UCI Adult

Test conditional independence on the Adult income dataset, with missingness imposed on education columns.

```python
from citest.data import adult

dataset = adult(n=1000, ci=True, mcar_prop=0.5)

test = CIMissTest(
    dataset,
    m=10,
    n_folds=10,
    classifier_args={"n_estimators": 20, "n_jobs": 8},
)
test.run()
test.summary()
```

The `adult` DGP downloads the UCI Adult dataset and applies controlled MAR missingness. Set `ci=False` to impose outcome-dependent missingness instead.

## 4. Custom imputer and classifier

Swap in a different imputer and classifier:

```python
from citest import CIMissTest
from citest.data import MAR1
from citest.imputer import IterativeImputer
from citest.classifier import LogisticClassifier

dataset = MAR1(n=500, ci=False)

test = CIMissTest(
    dataset,
    imputer=IterativeImputer,
    classifier=LogisticClassifier,
    m=10,
    n_folds=10,
    imputer_args={"max_iter": 20},
)
test.run()
test.summary()
```

The `IterativeImputer` is faster than the default MIDAS imputer and works well for moderate-sized numeric data. `LogisticClassifier` assumes a linear relationship between features and missingness.

## 5. Kappa calibration

Use the kappa diagnostic to assess potential imputation bias:

```python
from citest import kappa_calibration_table, print_calibration_pivot

# Generate a full calibration table
table = kappa_calibration_table()
print(table.head(10))

# View a pivot for a fixed beta_yx
pivot = print_calibration_pivot(beta_yx=0.3)
print(pivot)
```

The pivot table shows kappa values with R-squared as rows and gamma as columns, making it easy to assess whether imputation bias is a concern for your data.

You can also compute kappa for specific parameter values:

```python
from citest import compute_kappa

kappa = compute_kappa(r2_x_z=0.5, beta_yx=0.3, gamma_x=0.2)
print(f"kappa = {kappa:.4f}")
```

Small absolute values of kappa (e.g. < 0.05) suggest that imputation bias is unlikely to affect the test result.
