#' Install the citest Python backend
#'
#' Creates an isolated Python environment and installs the `midasverse-citest-api`
#' package (which pulls in `midasverse-citest` as a dependency).
#'
#' This is the **only** function in the package that uses `reticulate`, and
#' only for environment creation. It is never used at runtime.
#'
#' @param method Character. One of `"pip"`, `"conda"`, or `"uv"`.
#' @param envname Character. Name of the virtual environment to create
#'   (default `"citest_env"`).
#' @param package Character. Package specifier to install
#'   (default `"midasverse-citest-api"`).
#'
#' @return No return value, called for side effects.
#'
#' @examples
#' \dontrun{
#' install_backend()
#' install_backend(method = "conda")
#' }
#' @export
install_backend <- function(method = c("pip", "conda", "uv"),
                            envname = "citest_env",
                            package = "midasverse-citest-api") {
  method <- match.arg(method)

  if (method == "pip") {
    rlang::check_installed("reticulate",
      reason = "to create a Python environment for citest"
    )
    if (!reticulate::virtualenv_exists(envname)) {
      reticulate::virtualenv_create(envname)
    }
    reticulate::virtualenv_install(envname, packages = package)
    venv_path <- file.path(reticulate::virtualenv_root(), envname)
    save_venv_path(venv_path)
    .pkg_env$venv <- normalizePath(venv_path, mustWork = FALSE)
    rlang::inform("Installed. The server will auto-detect this environment.")
  } else if (method == "conda") {
    rlang::check_installed("reticulate",
      reason = "to create a conda environment for citest"
    )
    if (!envname %in% reticulate::conda_list()$name) {
      reticulate::conda_create(envname)
    }
    reticulate::conda_install(envname, packages = package, pip = TRUE)
    py <- reticulate::conda_python(envname)
    venv_path <- dirname(dirname(py))
    save_venv_path(venv_path)
    .pkg_env$venv <- normalizePath(venv_path, mustWork = FALSE)
    rlang::inform("Installed via conda. The server will auto-detect this environment.")
  } else if (method == "uv") {
    env_path <- file.path(tempdir(), envname)
    status <- system2("uv", c("venv", env_path))
    if (status != 0L) {
      rlang::abort("Failed to create uv virtual environment.")
    }
    if (.Platform$OS.type == "windows") {
      pip <- file.path(env_path, "Scripts", "pip")
    } else {
      pip <- file.path(env_path, "bin", "pip")
    }
    status <- system2("uv", c("pip", "install", "--python", pip, package))
    if (status != 0L) {
      rlang::abort("Failed to install packages via uv.")
    }
    save_venv_path(env_path)
    .pkg_env$venv <- normalizePath(env_path, mustWork = FALSE)
    rlang::inform(paste0("Installed via uv at ", env_path,
                         ". The server will auto-detect this environment."))
  }

  invisible(NULL)
}


#' Update the citest Python backend
#'
#' Upgrades the `midasverse-citest-api` package (and its dependencies) in the
#' existing Python environment. Stops the running server first so that the
#' new version is loaded on next use.
#'
#' @param method Character. One of `"pip"`, `"conda"`, or `"uv"`.
#'   Must match the method used during installation.
#' @param envname Character. Name of the virtual environment
#'   (default `"citest_env"`).
#' @param package Character. Package specifier to upgrade
#'   (default `"midasverse-citest-api"`).
#'
#' @return No return value, called for side effects.
#'
#' @examples
#' \dontrun{
#' update_backend()
#' }
#' @export
update_backend <- function(method = c("pip", "conda", "uv"),
                           envname = "citest_env",
                           package = "midasverse-citest-api") {
  method <- match.arg(method)

  # Stop the server so the new version is picked up on next start
  try(stop_server(), silent = TRUE)

  venv <- .pkg_env$venv
  if (is.null(venv)) {
    rlang::abort(
      "No saved Python environment found. Run install_backend() first."
    )
  }

  if (.Platform$OS.type == "windows") {
    pip <- file.path(venv, "Scripts", "pip")
  } else {
    pip <- file.path(venv, "bin", "pip")
  }

  if (method == "pip") {
    status <- system2(pip, c("install", "--upgrade", package))
    if (status != 0L) {
      rlang::abort("pip install --upgrade failed.")
    }
  } else if (method == "conda") {
    rlang::check_installed("reticulate",
      reason = "to upgrade packages in a conda environment"
    )
    reticulate::conda_install(envname, packages = package, pip = TRUE)
  } else if (method == "uv") {
    status <- system2("uv", c("pip", "install", "--upgrade",
                               "--python", pip, package))
    if (status != 0L) {
      rlang::abort("uv pip install --upgrade failed.")
    }
  }

  rlang::inform(paste0("Updated ", package, " successfully."))
  invisible(NULL)
}


#' Uninstall the citest Python backend
#'
#' Stops the running server (if any), removes the Python environment created by
#' [install_backend()], and clears the saved configuration.
#'
#' @param envname Character. Name of the virtual environment to remove
#'   (default `"citest_env"`).
#' @param method Character. One of `"pip"`, `"conda"`, or `"uv"`.
#'   Must match the method used during installation.
#'
#' @return No return value, called for side effects.
#'
#' @examples
#' \dontrun{
#' uninstall_backend()
#' uninstall_backend(method = "conda")
#' }
#' @export
uninstall_backend <- function(method = c("pip", "conda", "uv"),
                              envname = "citest_env") {
  method <- match.arg(method)

  # Stop the server if it is running
  try(stop_server(), silent = TRUE)

  if (method == "pip") {
    rlang::check_installed("reticulate",
      reason = "to remove a Python virtual environment"
    )
    if (reticulate::virtualenv_exists(envname)) {
      reticulate::virtualenv_remove(envname, confirm = FALSE)
      rlang::inform(paste0("Removed virtualenv '", envname, "'."))
    } else {
      rlang::inform(paste0("Virtualenv '", envname, "' not found; nothing to remove."))
    }
  } else if (method == "conda") {
    rlang::check_installed("reticulate",
      reason = "to remove a conda environment"
    )
    if (envname %in% reticulate::conda_list()$name) {
      reticulate::conda_remove(envname)
      rlang::inform(paste0("Removed conda environment '", envname, "'."))
    } else {
      rlang::inform(paste0("Conda environment '", envname,
                           "' not found; nothing to remove."))
    }
  } else if (method == "uv") {
    env_path <- .pkg_env$venv
    if (!is.null(env_path) && dir.exists(env_path)) {
      unlink(env_path, recursive = TRUE)
      rlang::inform(paste0("Removed uv environment at '", env_path, "'."))
    } else {
      rlang::inform("No saved uv environment found; nothing to remove.")
    }
  }

  clear_venv_path()
  invisible(NULL)
}
