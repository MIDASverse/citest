# citestR

R client for the **citest** conditional-independence-of-missingness test.

`citestR` communicates with a local Python API server over HTTP, so no
`reticulate` dependency is needed at runtime. The package provides
functions to run the test, retrieve summaries, compute imputer
diagnostics, and generate calibration tables.

## Installation

Install from CRAN:

```r
install.packages("citestR")
```

Or install the development version from GitHub:

```r
# install.packages("remotes")
remotes::install_github("midasverse/citest", subdir = "citestR")
```

## Python backend setup

The package requires a Python (>= 3.9) environment with the
`midasverse-citest-api` package. You can set this up using the built-in
helper:

```r
library(citestR)
install_backend()
```

Or install manually:

```bash
pip install midasverse-citest-api
```

## Quick start

```r
library(citestR)

# Create some data with missing values
data <- data.frame(
  Y  = rnorm(500),
  X1 = rnorm(500),
  X2 = rnorm(500)
)
data$X1[sample(500, 50)] <- NA

# Run the conditional independence test
result <- ci_test(data, y = "Y")

# View a summary
get_summary(result$model_id)

# Stop the server when finished
stop_server()
```

## How it works

The conditional-independence-of-missingness test checks whether the
pattern of missing data in the explanatory variables is independent of
the outcome, conditional on the observed data. The test uses
multiply-imputed datasets and cross-validated classifiers to estimate a
test statistic and p-value. See `vignette("getting-started", package =
"citestR")` for a detailed walkthrough.

## License

MIT
