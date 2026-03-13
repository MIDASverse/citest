# citestR 0.1.0

* Initial CRAN release.

## Exported functions

### Server management
- `start_server()`: Launch the citest Python API server.
- `stop_server()`: Stop the running server.
- `ensure_server()`: Start the server if it is not already running.

### Dataset creation
- `make_dataset()`: Send a data frame to the server as a Dataset.
- `make_dataset_parquet()`: Upload a Parquet file to the server.
- `simulate_data()`: Generate simulated data via built-in DGPs.

### Testing
- `ci_test()`: Run the conditional-independence-of-missingness test.
- `get_summary()`: Retrieve a structured summary of test results.

### Diagnostics
- `imputer_r2()`: Estimate imputer out-of-sample R-squared.
- `compute_kappa()`: Compute theoretical imputation bias kappa.
- `kappa_calibration_table()`: Generate a kappa calibration table.
- `calibration_pivot()`: Generate a calibration pivot table.

### Installation
- `install_backend()`: Install the Python backend into an isolated environment.
- `update_backend()`: Upgrade the Python backend to the latest version.
- `uninstall_backend()`: Remove the Python backend environment.
