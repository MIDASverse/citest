# Imputers

Multiple imputation backends. Each imputer takes a `Dataset` and produces `m` completed datasets.

| Class | Backend | Best for |
|---|---|---|
| `MidasImputer` (default) | MIDAS denoising autoencoder | Mixed types; general purpose |
| `IterativeImputer` | scikit-learn `IterativeImputer` | Moderate-sized numeric data |
| `IterativeImputer2` | Robust `IterativeImputer` variant | Wide or sparse data |

## MidasImputer

::: citest.imputer.MidasImputer
    options:
      members:
        - get_m_complete

## IterativeImputer

::: citest.imputer.IterativeImputer
    options:
      members:
        - get_m_complete

## IterativeImputer2

::: citest.imputer.IterativeImputer2
    options:
      members:
        - get_m_complete
