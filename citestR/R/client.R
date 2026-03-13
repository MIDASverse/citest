# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#' Convert an R matrix / data.frame to a nested list suitable for JSON
#' @keywords internal
to_nested_list <- function(x) {
  x <- as.matrix(x)
  # Replace NaN/NA with NULL for JSON null
  lapply(seq_len(nrow(x)), function(i) {
    row <- unname(as.list(x[i, ]))
    lapply(row, function(v) if (is.na(v)) NULL else v)
  })
}

#' POST JSON and return parsed body
#' @keywords internal
post_json <- function(path, body, timeout = 300) {
  # Ensure empty lists serialize as {} (JSON object) not [] (JSON array)
  if (length(body) == 0L) body <- setNames(list(), character(0))
  resp <- base_req(path) |>
    httr2::req_body_json(body, auto_unbox = TRUE) |>
    httr2::req_timeout(timeout) |>
    httr2::req_perform()
  httr2::resp_body_json(resp, simplifyVector = TRUE)
}

#' GET and return parsed body
#' @keywords internal
get_json <- function(path, timeout = 30) {
  resp <- base_req(path) |>
    httr2::req_timeout(timeout) |>
    httr2::req_perform()
  httr2::resp_body_json(resp, simplifyVector = TRUE)
}


# ---------------------------------------------------------------------------
# Dataset endpoints
# ---------------------------------------------------------------------------

#' Create a dataset on the server
#'
#' Sends a data frame to the citest API server and creates a `Dataset` object.
#'
#' @param data A data frame (may contain `NA` for missing values).
#' @param y Character. Name of the outcome variable.
#' @param expl_vars Character vector of explanatory variable names, or `NULL`
#'   for all non-outcome columns.
#' @param onehot Logical. One-hot encode categorical columns (default `TRUE`).
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A list with elements `dataset_id`, `n`, `columns`, `y_name`,
#'   `expl_vars`, and `pct_missing`.
#'
#' @examplesIf citestR:::has_server()
#' df <- data.frame(Y = rnorm(100), X1 = rnorm(100))
#' ds <- make_dataset(df, y = "Y")
#' ds$dataset_id
#' @export
make_dataset <- function(data, y, expl_vars = NULL, onehot = TRUE, ...) {
  ensure_server(...)
  cols <- colnames(data)
  body <- list(
    data = to_nested_list(data),
    columns = cols,
    y = y,
    expl_vars = expl_vars,
    onehot = onehot
  )
  post_json("/dataset/make", body)
}

#' Create a dataset from a Parquet file
#'
#' Uploads a Parquet file to the citest API server.
#'
#' @param file Path to a `.parquet` file.
#' @param y Character. Name of the outcome variable.
#' @param expl_vars Character vector of explanatory variable names, or `NULL`.
#' @param onehot Logical. One-hot encode categorical columns (default `TRUE`).
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A list with elements `dataset_id`, `n`, `columns`, `y_name`,
#'   `expl_vars`, and `pct_missing`.
#'
#' @examplesIf citestR:::has_server()
#' ds <- make_dataset_parquet("data.parquet", y = "Y")
#' @export
make_dataset_parquet <- function(file, y, expl_vars = NULL, onehot = TRUE, ...) {
  ensure_server(...)
  ev <- if (!is.null(expl_vars)) paste(expl_vars, collapse = ",") else NULL
  resp <- base_req("/dataset/make_parquet") |>
    httr2::req_body_multipart(
      file = curl::form_file(file, type = "application/octet-stream"),
      y = y,
      expl_vars = ev,
      onehot = as.character(onehot)
    ) |>
    httr2::req_timeout(300) |>
    httr2::req_error(body = function(resp) {
      tryCatch({
        detail <- httr2::resp_body_json(resp)$detail
        if (is.character(detail)) detail else paste(detail, collapse = "; ")
      }, error = function(e) NULL)
    }) |>
    httr2::req_perform()
  httr2::resp_body_json(resp, simplifyVector = TRUE)
}


# ---------------------------------------------------------------------------
# Simulate
# ---------------------------------------------------------------------------

#' Generate a simulated dataset
#'
#' Calls one of the built-in data-generating processes on the Python server.
#'
#' @param dgp Character. Name of the DGP (e.g. `"single_mar"`, `"adult"`).
#' @param n Integer. Number of observations.
#' @param ci Logical. Conditional independence holds (`TRUE`) or not.
#' @param missing_mech Character. Missingness mechanism (`"linear"` or `"xor"`).
#' @param beta_y Numeric or `NULL`. Outcome effect size (for DGPs that use it).
#' @param mcar_prop Numeric or `NULL`. Proportion of MCAR missingness.
#' @param k Integer or `NULL`. Number of columns (for the `adult` DGP).
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A list with `dataset_id`, `n`, `columns`, `pct_missing`.
#'
#' @examplesIf citestR:::has_server()
#' sim <- simulate_data("single_mar", n = 500, ci = TRUE)
#' @export
simulate_data <- function(dgp, n = 1000L, ci = TRUE,
                          missing_mech = "linear",
                          beta_y = NULL, mcar_prop = NULL, k = NULL, ...) {
  ensure_server(...)
  body <- list(dgp = dgp, n = n, ci = ci, missing_mech = missing_mech)
  if (!is.null(beta_y)) body$beta_y <- beta_y
  if (!is.null(mcar_prop)) body$mcar_prop <- mcar_prop
  if (!is.null(k)) body$k <- k
  post_json("/simulate", body)
}


# ---------------------------------------------------------------------------
# Test lifecycle
# ---------------------------------------------------------------------------

#' Run the conditional independence test
#'
#' All-in-one convenience function: creates a dataset on the server, builds a
#' `CIMissTest`, runs it, and returns the results.
#'
#' @param data A data frame (may contain `NA`).
#' @param y Character. Name of the outcome variable.
#' @param expl_vars Character vector of explanatory variable names, or `NULL`.
#' @param onehot Logical. One-hot encode categoricals (default `TRUE`).
#' @param imputer Character. Imputer backend: `"midas"` (default),
#'   `"iterative"`, `"iterative2"`, `"complete"`, or `"null"`.
#' @param classifier Character. Classifier backend: `"rf"` (default),
#'   `"et"`, or `"logistic"`.
#' @param m Integer. Number of multiply-imputed datasets (default 10).
#' @param n_folds Integer. Number of cross-validation folds (default 10).
#' @param classifier_args Named list of extra classifier arguments.
#' @param imputer_args Named list of extra imputer arguments.
#' @param random_state Integer. Random seed (default 42).
#' @param target_level Character. `"variable"` or `"column"`.
#' @param variance_method Character. `"mi_crossfit"` or `"legacy_fold"`.
#' @param subsample_cap Integer or `NULL`. Maximum rows to subsample.
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A list with elements `model_id`, `dataset_id`, and `results`.
#'   The `results` element contains `m`, `B`, `W_bar`, `T`, `t_k`, `p_k`,
#'   `p_2s`, and optionally `df`.
#'
#' @examplesIf citestR:::has_server()
#' df <- data.frame(Y = rnorm(200), X1 = rnorm(200), X2 = rnorm(200))
#' df$X1[sample(200, 20)] <- NA
#' result <- ci_test(df, y = "Y")
#' result$results$p_2s
#' @export
ci_test <- function(data, y, expl_vars = NULL, onehot = TRUE,
                    imputer = "midas", classifier = "rf",
                    m = 10L, n_folds = 10L,
                    classifier_args = list(),
                    imputer_args = list(),
                    random_state = 42L,
                    target_level = "variable",
                    variance_method = "mi_crossfit",
                    subsample_cap = 2000L,
                    ...) {
  ensure_server(...)

  nrows <- nrow(data)
  use_parquet <- FALSE

  # If arrow is available and data is large, use the parquet path

  if (nrows > 100000L && rlang::is_installed("arrow")) {
    use_parquet <- TRUE
  }

  if (use_parquet) {
    tmp <- tempfile(fileext = ".parquet")
    on.exit(unlink(tmp), add = TRUE)
    arrow::write_parquet(data, tmp)
    ev <- if (!is.null(expl_vars)) paste(expl_vars, collapse = ",") else NULL
    resp <- base_req("/fit_parquet") |>
      httr2::req_body_multipart(
        file = curl::form_file(tmp, type = "application/octet-stream"),
        y = y,
        expl_vars = ev,
        onehot = as.character(onehot),
        imputer = imputer,
        classifier = classifier,
        m = as.character(m),
        n_folds = as.character(n_folds),
        random_state = as.character(random_state),
        target_level = target_level,
        variance_method = variance_method,
        subsample_cap = if (!is.null(subsample_cap)) as.character(subsample_cap) else ""
      ) |>
      httr2::req_timeout(300) |>
      httr2::req_error(body = function(resp) {
      tryCatch({
        detail <- httr2::resp_body_json(resp)$detail
        if (is.character(detail)) detail else paste(detail, collapse = "; ")
      }, error = function(e) NULL)
    }) |>
      httr2::req_perform()
    out <- httr2::resp_body_json(resp, simplifyVector = TRUE)
    class(out) <- "citest_result"
    return(out)
  }

  cols <- colnames(data)
  body <- list(
    data = to_nested_list(data),
    columns = cols,
    y = y,
    expl_vars = expl_vars,
    onehot = onehot,
    imputer = imputer,
    classifier = classifier,
    m = m,
    n_folds = n_folds,
    random_state = random_state,
    target_level = target_level,
    variance_method = variance_method,
    subsample_cap = subsample_cap
  )
  if (length(classifier_args) > 0L) body$classifier_args <- classifier_args
  if (length(imputer_args) > 0L) body$imputer_args <- imputer_args
  out <- post_json("/fit", body, timeout = 300)
  class(out) <- "citest_result"
  out
}


#' Get a summary of test results
#'
#' Retrieves a structured summary for a previously fitted model.
#'
#' @param model_id Character. UUID returned by [ci_test()].
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A list with elements `outcome`, `imputer`, `classifier`,
#'   `variance_method`, `mean_difference`, `t_statistic`, `df`, `p_value`,
#'   and `p_value_two_sided`.
#'
#' @examplesIf citestR:::has_server()
#' result <- ci_test(df, y = "Y")
#' get_summary(result$model_id)
#' @export
get_summary <- function(model_id, ...) {
  ensure_server(...)
  out <- get_json(paste0("/test/", model_id, "/summary"))
  class(out) <- "citest_summary"
  out
}

#' Estimate imputer out-of-sample R-squared
#'
#' Runs a mask-and-impute diagnostic on the server.
#'
#' @param model_id Character. UUID returned by [ci_test()].
#' @param mask_frac Numeric. Fraction of observed cells to hold out (default 0.2).
#' @param m_eval Integer. Number of imputations to average over (default 1).
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A list with `mean_r2` and `per_variable` (named numeric vector).
#'
#' @examplesIf citestR:::has_server()
#' result <- ci_test(df, y = "Y")
#' imputer_r2(result$model_id)
#' @export
imputer_r2 <- function(model_id, mask_frac = 0.2, m_eval = 1L, ...) {
  ensure_server(...)
  body <- list(mask_frac = mask_frac, m_eval = m_eval)
  post_json(paste0("/test/", model_id, "/imputer_r2"), body, timeout = 300)
}


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

#' Compute theoretical imputation bias kappa
#'
#' @param r2_x_z Numeric. R-squared of X on observed covariates Z.
#' @param beta_yx Numeric. Coefficient of X in the Y equation.
#' @param gamma_x Numeric. Loading of X in the missingness equation.
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A single numeric value (kappa).
#'
#' @examplesIf citestR:::has_server()
#' compute_kappa(r2_x_z = 0.5, beta_yx = 0.3, gamma_x = 0.2)
#' @export
compute_kappa <- function(r2_x_z, beta_yx, gamma_x, ...) {
  ensure_server(...)
  body <- list(r2_x_z = r2_x_z, beta_yx = beta_yx, gamma_x = gamma_x)
  res <- post_json("/compute_kappa", body)
  res$kappa
}

#' Generate a kappa calibration table
#'
#' @param r2_grid Numeric vector of R-squared values, or `NULL` for defaults.
#' @param beta_grid Numeric vector of beta values, or `NULL` for defaults.
#' @param gamma_grid Numeric vector of gamma values, or `NULL` for defaults.
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A data frame with columns `r2_x_z`, `beta_yx`, `gamma_x`, `kappa`,
#'   `abs_kappa`.
#'
#' @examplesIf citestR:::has_server()
#' kappa_calibration_table(r2_grid = c(0.3, 0.5, 0.7))
#' @export
kappa_calibration_table <- function(r2_grid = NULL, beta_grid = NULL,
                                    gamma_grid = NULL, ...) {
  ensure_server(...)
  body <- list()
  if (!is.null(r2_grid)) body$r2_grid <- r2_grid
  if (!is.null(beta_grid)) body$beta_grid <- beta_grid
  if (!is.null(gamma_grid)) body$gamma_grid <- gamma_grid
  res <- post_json("/kappa_calibration_table", body)
  mat <- if (is.matrix(res$data)) res$data else do.call(rbind, res$data)
  df <- as.data.frame(mat)
  colnames(df) <- res$columns
  df
}

#' Generate a calibration pivot table
#'
#' Rows are R-squared values, columns are gamma_x values, for a fixed beta_yx.
#'
#' @param beta_yx Numeric. Fixed beta_yx value (default 0.3).
#' @param r2_grid Numeric vector, or `NULL`.
#' @param beta_grid Numeric vector, or `NULL`.
#' @param gamma_grid Numeric vector, or `NULL`.
#' @param ... Arguments forwarded to [ensure_server()].
#'
#' @return A data frame (pivot table).
#'
#' @examplesIf citestR:::has_server()
#' calibration_pivot(beta_yx = 0.3)
#' @export
calibration_pivot <- function(beta_yx = 0.3, r2_grid = NULL,
                              beta_grid = NULL, gamma_grid = NULL, ...) {
  ensure_server(...)
  body <- list(beta_yx = beta_yx)
  if (!is.null(r2_grid)) body$r2_grid <- r2_grid
  if (!is.null(beta_grid)) body$beta_grid <- beta_grid
  if (!is.null(gamma_grid)) body$gamma_grid <- gamma_grid
  res <- post_json("/print_calibration_pivot", body)
  mat <- if (is.matrix(res$data)) res$data else do.call(rbind, res$data)
  df <- as.data.frame(mat)
  colnames(df) <- res$columns
  rownames(df) <- res$index
  df
}
