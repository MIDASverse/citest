# Classifiers

Per-column probabilistic classifiers used to predict missingness indicators. Each classifier wraps a scikit-learn estimator and fits a separate model per target column.

| Class | Estimator | Notes |
|---|---|---|
| `RFClassifier` (default) | Random forest | Auto-tunes `max_features` and `min_samples_leaf` |
| `ETClassifier` | Extra trees | Faster training; more variance |
| `LogisticClassifier` | Logistic regression | Assumes linear relationships |

## RFClassifier

::: citest.classifier.RFClassifier

## ETClassifier

::: citest.classifier.ETClassifier

## LogisticClassifier

::: citest.classifier.LogisticClassifier
