#' Print a citest result
#'
#' Displays a concise summary of the conditional independence test result,
#' including the test statistic, degrees of freedom, p-value, and a plain
#' language interpretation.
#'
#' @param x A `citest_result` object returned by [ci_test()].
#' @param ... Additional arguments (currently ignored).
#'
#' @return Invisibly returns `x`.
#'
#' @examples
#' result <- structure(list(
#'   model_id = "example-id",
#'   dataset_id = "example-ds",
#'   results = list(m = 0.12, t_k = 2.5, df = 9, p_2s = 0.034)
#' ), class = "citest_result")
#' print(result)
#'
#' @export
print.citest_result <- function(x, ...) {
  r <- x$results
  cat("\n  Conditional Independence of Missingness Test\n\n")
  cat("  Mean diff in BCE:", format(r$m, digits = 4), "\n")
  cat("  t-statistic:    ", format(r$t_k, digits = 4), "\n")
  if (!is.null(r$df)) {
    cat("  df:             ", format(r$df, digits = 4), "\n")
  }
  cat("  p-value:        ", format(r$p_2s, digits = 4), "(two-sided)\n")
  cat("\n")
  if (!is.na(r$p_2s)) {
    if (r$p_2s < 0.001) {
      cat("  Result: Strong evidence against conditional independence (p < 0.001)\n")
    } else if (r$p_2s < 0.01) {
      cat("  Result: Evidence against conditional independence (p < 0.01)\n")
    } else if (r$p_2s < 0.05) {
      cat("  Result: Evidence against conditional independence (p < 0.05)\n")
    } else if (r$p_2s < 0.1) {
      cat("  Result: Weak evidence against conditional independence (p < 0.10)\n")
    } else {
      cat("  Result: No evidence against conditional independence\n")
    }
  }
  cat("\n")
  invisible(x)
}

#' Print a citest summary
#'
#' Displays a formatted summary of a fitted conditional independence test,
#' including model configuration and key results.
#'
#' @param x A `citest_summary` object returned by [get_summary()].
#' @param ... Additional arguments (currently ignored).
#'
#' @return Invisibly returns `x`.
#'
#' @examples
#' smry <- structure(list(
#'   outcome = "Y",
#'   imputer = "midas",
#'   classifier = "rf",
#'   variance_method = "mi_crossfit",
#'   mean_difference = 0.12,
#'   t_statistic = 2.5,
#'   df = 9,
#'   p_value_two_sided = 0.034
#' ), class = "citest_summary")
#' print(smry)
#'
#' @export
print.citest_summary <- function(x, ...) {
  cat("\n  CI Test Summary\n\n")
  cat("  Outcome:         ", x$outcome, "\n")
  cat("  Imputer:         ", x$imputer, "\n")
  cat("  Classifier:      ", x$classifier, "\n")
  cat("  Variance method: ", x$variance_method, "\n")
  cat("\n")
  cat("  Mean difference: ", format(x$mean_difference, digits = 4), "\n")
  cat("  t-statistic:     ", format(x$t_statistic, digits = 4), "\n")
  if (!is.null(x$df)) {
    cat("  df:              ", format(x$df, digits = 4), "\n")
  }
  cat("  p-value:         ", format(x$p_value_two_sided, digits = 4),
      "(two-sided)\n")
  cat("\n")
  invisible(x)
}
