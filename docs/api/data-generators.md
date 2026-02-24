# Data Generators

Functions that produce `Dataset` objects with controlled missingness for simulation studies. Each function returns a populated `Dataset` with `full_data` set for evaluation.

All generators accept `ci=True` (conditional independence holds) or `ci=False` (outcome influences missingness).

## Synthetic DGPs

Simple linear data-generating processes for controlled experiments.

| Function | Missingness | Description |
|---|---|---|
| `single_mar` | MAR | Simple linear DGP, missingness on X2 |
| `single_mnar` | MNAR | Linear DGP, missingness depends on unobserved Z |
| `MAR1` | MAR | King (2001) DGP with multi-variable missingness |
| `MNAR1` | MNAR | MNAR variant of King (2001) DGP |

::: citest.data.single_mar

::: citest.data.single_mnar

::: citest.data.MAR1

::: citest.data.MNAR1

## Real-data DGPs

Download real datasets and impose controlled missingness mechanisms.

| Function | Source | Default *n* |
|---|---|---|
| `adult` | UCI Adult (census income) | 1000 |
| `adult_mnar` | UCI Adult with MNAR via unobserved sex | 1000 |
| `mushrooms` | UCI Mushroom | 1000 |
| `breast_cancer` | Wisconsin Breast Cancer | 500 |
| `wine` | UCI Wine | 500 |
| `diabetes` | Diabetes progression | 442 |
| `covertype` | Covertype | 5000 |
| `california_housing` | California housing | -- |
| `german_credit` | German credit | -- |
| `bank_marketing` | Bank marketing | -- |
| `ames_housing` | Ames housing | -- |
| `give_me_some_credit` | Give Me Some Credit | -- |

::: citest.data.adult

::: citest.data.adult_mnar

::: citest.data.mushrooms

::: citest.data.breast_cancer

::: citest.data.wine

::: citest.data.diabetes

::: citest.data.covertype

::: citest.data.california_housing

::: citest.data.german_credit

::: citest.data.bank_marketing

::: citest.data.ames_housing

::: citest.data.give_me_some_credit
