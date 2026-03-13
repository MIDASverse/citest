## R CMD check results

0 errors | 0 warnings | 1 note

* This is a new submission.

## Notes for CRAN reviewers

### Method DOI link

There is no DOI link for this method yet. We will add this shortly in an upcoming version.

### Python SystemRequirement

This package communicates with a local Python FastAPI server over HTTP.
The Python backend (`midasverse-citest-api`) must be installed separately.
No `reticulate` dependency is needed at runtime; `reticulate` is used
only by the optional `install_backend()` helper.

### Examples

Most exported functions require a running Python backend server that is
not available on CRAN check infrastructure.

- **`@examplesIf`** is used for functions that call the Python server
  (`ci_test`, `make_dataset`, `simulate_data`, `start_server`, etc.).
  Examples are guarded by `citestR:::has_server()` and only run when
  the backend is available. They are skipped during `R CMD check` but
  runnable by users who have the backend installed.
- **`\dontrun{}`** is used only for the installation helpers
  (`install_backend`, `update_backend`, `uninstall_backend`) because
  they create or delete Python virtual environments (filesystem
  side-effects).
- The `print` methods have fully runnable examples that execute on every
  platform without any external dependencies.

### Vignette

The `getting-started` vignette executes code during build using mocked
HTTP responses (via `httr2::with_mocked_responses()`), so the package
interface is tested without requiring a live Python backend.

### Tests

Unit tests use `httr2::with_mocked_responses()` to mock all HTTP
communication. Tests that would require a live Python server are
skipped on CRAN via `skip_on_cran()`.

## Test environments

* macOS Sequoia 15.7.3 (local), R 4.5.2
* r-devel-macosx-arm64, R 4.6.0
* Windows, R 4.5.2
