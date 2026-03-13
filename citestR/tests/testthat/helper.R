# Mock response helpers for httr2::with_mocked_responses()

#' Build a mock response function that returns fixed JSON
#' @param body Named list to serialize as JSON.
#' @param status HTTP status code (default 200).
#' @keywords internal
mock_json_response <- function(body, status = 200L) {
  function(req) {
    httr2::response(
      status_code = status,
      headers = list("Content-Type" = "application/json"),
      body = charToRaw(jsonlite::toJSON(body, auto_unbox = TRUE))
    )
  }
}

#' Force .pkg_env into a state where the server appears running
#' @keywords internal
fake_server_running <- function(port = 9999L) {
  env <- citestR:::.pkg_env
  # Use a dummy process-like object

  env$process <- list(is_alive = function() TRUE)

  env$port <- port
  env$base_url <- paste0("http://127.0.0.1:", port)
}

#' Reset .pkg_env after tests
#' @keywords internal
reset_server_state <- function() {
  env <- citestR:::.pkg_env
  env$process <- NULL
  env$port <- NULL
  env$base_url <- NULL
}
