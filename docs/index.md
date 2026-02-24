# citest

**Conditional independence testing for missing data.**

`citest` tests whether an outcome variable is independent of missingness, conditional on the observed covariates. It compares classifier performance in predicting missingness with and without the outcome, using multiple imputation and cross-fitting to produce a valid *t*-statistic and *p*-value.

```python
import pandas as pd
from citest import CIMissTest
from citest.data import Dataset

data = pd.read_csv("my_data.csv")
ds = Dataset()
ds.make(data, y="outcome", expl_vars=["x1", "x2", "x3"])

test = CIMissTest(ds)
test.run()
test.summary()
```

## Install

```bash
pip install citest
```

## Next steps

- [Getting Started](getting-started.md) -- install, prepare data, and run your first test
- [How It Works](methodology.md) -- the statistical methodology behind the test
- [Examples](examples.md) -- worked examples with simulated and real data
- [API Reference](api/cimisstest.md) -- full class and function documentation
