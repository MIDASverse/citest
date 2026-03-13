#' Find a free TCP port
#'
#' Samples random ports in the dynamic range and uses [serverSocket()] to
#' verify availability.
#'
#' @return Integer port number.
#' @keywords internal
find_free_port <- function() {
  for (i in seq_len(20L)) {
    port <- sample(49152L:65535L, 1L)
    tryCatch(
      {
        srv <- serverSocket(port = port)
        close(srv)
        return(port)
      },
      error = function(e) NULL
    )
  }
  rlang::abort("Could not find a free port after 20 attempts.")
}

#' Start the citest API server
#'
#' Launches `python -m citest_api` as a background process and waits for the
#' `/health` endpoint to respond.
#'
#' @param python Path to the Python interpreter (default `"python3"`).
#' @param port Port to bind to. If `NULL`, a free port is chosen automatically.
#' @param venv Path to a Python virtual environment.
#'   If supplied, the interpreter is taken from `<venv>/bin/python`
#'   (or `<venv>/Scripts/python.exe` on Windows).
#' @param max_wait Maximum number of 0.5-second polling attempts (default 120,
#'   i.e. 60 seconds). The first launch may be slower due to Python import
#'   caching.
#'
#' @return Invisibly returns the port number.
#'
#' @examplesIf citestR:::has_server()
#' start_server()
#' start_server(venv = "~/.virtualenvs/citest_env")
#' @export
start_server <- function(python = "python3", port = NULL, venv = NULL,
                         max_wait = 120L) {
  if (!is.null(.pkg_env$process) && .pkg_env$process$is_alive()) {
    rlang::inform("Server is already running.")
    return(invisible(.pkg_env$port))
  }

  # Use saved venv from install_backend() when none is passed explicitly
  if (is.null(venv) && !is.null(.pkg_env$venv)) {
    venv <- .pkg_env$venv
  }

  if (!is.null(venv)) {
    venv <- path.expand(venv)
    if (.Platform$OS.type == "windows") {
      python <- file.path(venv, "Scripts", "python.exe")
    } else {
      python <- file.path(venv, "bin", "python")
    }
  }

  if (is.null(port)) {
    port <- find_free_port()
  }

  # Write stderr to a temp file to avoid a pipe-buffer deadlock
  # (uvicorn logs heavily to stderr).
  stderr_file <- tempfile("citest_stderr_")
  proc <- processx::process$new(
    command = python,
    args = c("-m", "citest_api", "--port", as.character(port)),
    stdout = NULL,
    stderr = stderr_file,
    cleanup_tree = TRUE
  )

  .pkg_env$process <- proc
  .pkg_env$port <- port
  .pkg_env$base_url <- paste0("http://127.0.0.1:", port)

  # Poll /health until the server is ready
  rlang::inform("Starting Python server...")
  ready <- FALSE
  for (i in seq_len(max_wait)) {
    Sys.sleep(0.5)
    if (i %% 10L == 0L) {
      rlang::inform(paste0("  Waiting for server... (", i * 0.5, "s)"))
    }
    if (!proc$is_alive()) {
      err <- readLines(stderr_file, warn = FALSE)
      unlink(stderr_file)
      rlang::abort(c(
        "Python server process died during startup.",
        paste(err, collapse = "\n")
      ))
    }
    tryCatch(
      {
        resp <- httr2::request(.pkg_env$base_url) |>
          httr2::req_url_path_append("health") |>
          httr2::req_timeout(2) |>
          httr2::req_perform()
        if (httr2::resp_status(resp) == 200L) {
          ready <- TRUE
          break
        }
      },
      error = function(e) NULL
    )
  }

  if (!ready) {
    stop_server()
    rlang::abort("Server did not become ready within the timeout period.")
  }

  .pkg_env$python <- python
  rlang::inform(paste0("citest server running on port ", port))
  check_backend_version(python)
  invisible(port)
}

#' Check whether the installed backend is up-to-date with PyPI
#'
#' Compares the locally installed version of `midasverse-citest-api` against
#' the latest release on PyPI.
#' Runs silently on success; emits a message when an update is available.
#' Failures (e.g. no network) are silently ignored.
#'
#' @param python Path to the Python interpreter.
#' @param package PyPI package name (default `"midasverse-citest-api"`).
#' @return No return value, called for side effects.
#' @keywords internal
check_backend_version <- function(python,
                                  package = "midasverse-citest-api") {
  tryCatch({
    installed <- suppressWarnings(system2(
      python,
      c("-c", shQuote(paste0(
        "import importlib.metadata; ",
        "print(importlib.metadata.version('", package, "'))"
      ))),
      stdout = TRUE, stderr = FALSE
    ))
    if (!is.null(attr(installed, "status"))) return(invisible(NULL))
    installed <- trimws(installed)

    pypi_url <- paste0("https://pypi.org/pypi/", package, "/json")
    resp <- httr2::request(pypi_url) |>
      httr2::req_timeout(5) |>
      httr2::req_perform()
    latest <- httr2::resp_body_json(resp)$info$version

    if (!identical(installed, latest)) {
      rlang::inform(paste0(
        "A newer version of ", package, " is available (",
        installed, " -> ", latest, "). ",
        "Run update_backend() to upgrade."
      ))
    }
  }, error = function(e) NULL)
}

#' Stop the citest API server
#'
#' Kills the background Python process and clears the internal state.
#'
#' @return No return value, called for side effects.
#'
#' @examplesIf citestR:::has_server()
#' stop_server()
#' @export
stop_server <- function() {
  proc <- .pkg_env$process
  if (!is.null(proc)) {
    try(proc$kill_tree(), silent = TRUE)
  }
  .pkg_env$process <- NULL
  .pkg_env$port <- NULL
  .pkg_env$base_url <- NULL
  invisible(NULL)
}

#' Ensure the server is running
#'
#' Starts the server if it is not already running.
#' Called internally by every client function so users never have to manage
#' the server manually.
#'
#' @param ... Arguments forwarded to [start_server()].
#'
#' @return Invisibly returns the base URL of the running server.
#'
#' @examplesIf citestR:::has_server()
#' ensure_server()
#' @export
ensure_server <- function(...) {
  if (is.null(.pkg_env$process) || !.pkg_env$process$is_alive()) {
    start_server(...)
  }
  invisible(.pkg_env$base_url)
}

#' Build a base request pointing at the running server
#' @param path API path (e.g. "/fit").
#' @return An httr2 request object.
#' @keywords internal
base_req <- function(path) {
  if (is.null(.pkg_env$base_url)) {
    rlang::abort("Server is not running. Call start_server() or ensure_server().")
  }
  httr2::request(.pkg_env$base_url) |>
    httr2::req_url_path(path) |>
    httr2::req_error(body = function(resp) {
      tryCatch(
        {
          detail <- httr2::resp_body_json(resp)$detail
          if (is.character(detail)) detail else paste(detail, collapse = "; ")
        },
        error = function(e) NULL
      )
    })
}
