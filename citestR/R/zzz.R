#' @importFrom curl form_file
#' @importFrom httr2 request req_body_json req_body_multipart req_error
#'   req_perform req_timeout req_url_path req_url_path_append resp_body_json
#'   resp_body_string resp_status
#' @importFrom processx process
#' @importFrom rlang abort inform is_installed
#' @importFrom stats setNames
#' @keywords internal
"_PACKAGE"

# Private environment shared across the package
.pkg_env <- new.env(parent = emptyenv())

#' Path to the package config directory
#' @return Character path to the config directory.
#' @keywords internal
config_dir <- function() {
  tools::R_user_dir("citestR", "config")
}

#' Save the virtualenv path to persistent config
#' @param path Character path to save.
#' @return No return value, called for side effects.
#' @keywords internal
save_venv_path <- function(path) {
  dir <- config_dir()
  if (!dir.exists(dir)) dir.create(dir, recursive = TRUE)
  writeLines(normalizePath(path, mustWork = FALSE), file.path(dir, "venv_path"))
}

#' Load the saved virtualenv path (or NULL)
#' @return Character path or `NULL`.
#' @keywords internal
load_venv_path <- function() {
  f <- file.path(config_dir(), "venv_path")
  if (file.exists(f)) readLines(f, n = 1L) else NULL
}

#' Remove the saved virtualenv path
#' @return No return value, called for side effects.
#' @keywords internal
clear_venv_path <- function() {
  f <- file.path(config_dir(), "venv_path")
  if (file.exists(f)) unlink(f)
  .pkg_env$venv <- NULL
}

#' Check whether the citest server is running
#'
#' Returns `TRUE` if the package's background server process is alive.
#' Used as the guard for `@examplesIf` so that examples requiring the
#' Python backend are skipped when no server is available.
#'
#' @return Logical.
#' @keywords internal
has_server <- function() {
  !is.null(.pkg_env$process) && .pkg_env$process$is_alive()
}

.onLoad <- function(libname, pkgname) {
  .pkg_env$process <- NULL
  .pkg_env$port <- NULL
  .pkg_env$base_url <- NULL
  .pkg_env$venv <- load_venv_path()
}

.onUnload <- function(libpath) {
  try(stop_server(), silent = TRUE)
}
