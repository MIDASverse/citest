# How It Works

This page describes the statistical methodology behind the conditional independence test implemented in `citest`.

## The hypothesis

Let \(Y\) be the outcome variable, \(X\) the covariates, and \(R\) the missingness indicator (1 = observed, 0 = missing). The null hypothesis is:

\[
H_0: R \perp\!\!\!\perp Y \mid X
\]

That is, missingness is **conditionally independent** of the outcome given the covariates. Rejecting \(H_0\) provides evidence of MNAR-type missingness -- the outcome influences whether data are missing, even after conditioning on the observed covariates.

## Test procedure

The test has four stages.

### 1. Multiple imputation

The missing values in the dataset are multiply imputed to produce \(m\) completed datasets \(\{D^{(1)}, \ldots, D^{(m)}\}\). By default, `citest` uses the MIDAS denoising autoencoder, but iterative imputation is also supported.

### 2. Classifier comparison

For each imputed dataset, two classifiers are trained to predict the missingness indicator \(R\):

- **Full model**: uses the outcome \(Y\) and covariates \(X\) as features
- **Null model**: uses a **permuted** copy of \(Y\) (destroying any dependence on \(R\)) and covariates \(X\)

Both classifiers produce probabilistic predictions, which are scored using the **binary cross-entropy** (BCE) loss:

\[
\text{BCE}(p, r) = -\bigl[r \log p + (1 - r) \log(1 - p)\bigr]
\]

where \(p\) is the predicted probability and \(r \in \{0, 1\}\) is the true missingness indicator. The observation-level score is the weighted difference:

\[
g_i = \sum_j w_j \bigl[\text{BCE}(\hat{p}_{ij}^{\text{null}}, r_{ij}) - \text{BCE}(\hat{p}_{ij}^{\text{full}}, r_{ij})\bigr]
\]

where \(j\) indexes target columns and \(w_j\) are column-level weights proportional to the variance of missingness \(w_j \propto \hat{m}_j(1 - \hat{m}_j)\).

Under \(H_0\), the full and null models have the same expected loss, so \(\mathbb{E}[g_i] = 0\). Under the alternative, the full model has lower BCE (the outcome helps predict missingness), yielding \(\mathbb{E}[g_i] > 0\).

### 3. Cross-fitting

Predictions are made **out-of-fold** using \(K\)-fold cross-validation. For each fold \(k\):

1. The imputer is fit on the training fold
2. Both classifiers are trained on the training fold
3. Scores \(g_i\) are computed on the held-out fold

This avoids data leakage from using the same observations for fitting and evaluation. The per-imputation estimate of the test quantity is the mean of fold-level means:

\[
\hat{Q}^{(b)} = \frac{1}{K} \sum_{k=1}^{K} \bar{g}_k^{(b)}
\]

### 4. Test statistic

Estimates are combined across imputations using **Rubin's rules**.

The **within-imputation variance** accounts for sampling variability in the cross-fitting procedure, with a Nadeau--Bengio correction for the dependence induced by overlapping training sets:

\[
\hat{U}^{(b)} = \left(\frac{1}{K} + \frac{n_{\text{test}}}{n_{\text{train}}}\right) \cdot \frac{1}{K-1} \sum_{k=1}^{K} \bigl(\bar{g}_k^{(b)} - \hat{Q}^{(b)}\bigr)^2
\]

The **between-imputation variance** captures the additional uncertainty due to the missing data:

\[
B = \frac{1}{m-1} \sum_{b=1}^{m} \bigl(\hat{Q}^{(b)} - \bar{Q}\bigr)^2
\]

The total variance is:

\[
T = \bar{U} + \left(1 + \frac{1}{m}\right) B
\]

The test statistic follows a *t*-distribution:

\[
t = \frac{\bar{Q}}{\sqrt{T}} \sim t_\nu
\]

where the degrees of freedom \(\nu\) are computed using the **Barnard--Rubin** adjustment, which accounts for both the finite number of imputations and the finite complete-data degrees of freedom:

\[
\nu = \frac{\nu_{\text{old}} \cdot \nu_{\text{obs}}}{\nu_{\text{old}} + \nu_{\text{obs}}}
\]

with \(\nu_{\text{old}} = (m-1)(1 + 1/r)^2\) where \(r = (1 + 1/m)B / \bar{U}\), and \(\nu_{\text{obs}}\) depends on the complete-data degrees of freedom (\(K - 1\)).

## Variance methods

`citest` offers two variance estimation strategies via the `variance_method` parameter.

### `mi_crossfit` (default)

The recommended method. Computes observation-level scores and uses fold-level aggregation with the Nadeau--Bengio correction and Barnard--Rubin degrees of freedom, as described above.

Key properties:

- Uses **common random numbers** across imputations within each fold (same classifier seed and permutation), so between-imputation variance \(B\) reflects imputation uncertainty rather than Monte Carlo noise
- Returns Barnard--Rubin degrees of freedom alongside the test statistic

### `legacy_fold`

The original method. Computes a single scalar difference per fold per imputation, then applies Rubin's rules to the fold-level means. Uses \(K - 1\) degrees of freedom for the *t*-distribution.

## Kappa diagnostic

The `compute_kappa` function provides a theoretical measure of how much imputation bias might affect the test. It is defined as:

\[
\kappa = \frac{\gamma_x \cdot \beta_{yx} \cdot (1 - R^2_{X|Z})}{1 + \beta_{yx}^2 \cdot (1 - R^2_{X|Z})}
\]

where:

- \(R^2_{X|Z}\) is the R-squared of the partially-observed variable on the fully-observed covariates
- \(\beta_{yx}\) is the coefficient of the partially-observed variable in the outcome equation
- \(\gamma_x\) is the loading of the partially-observed variable in the missingness equation

Small absolute values of \(\kappa\) indicate that imputation bias is unlikely to distort the test. The `kappa_calibration_table` function generates a grid of \(\kappa\) values over realistic parameter ranges to help assess this.
