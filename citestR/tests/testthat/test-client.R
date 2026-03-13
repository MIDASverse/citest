test_that("ci_test sends correct request and parses response", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(
    model_id = "abc-123",
    dataset_id = "ds-456",
    results = list(
      m = 0.05,
      B = 0.01,
      W_bar = 0.02,
      `T` = 0.03,
      t_k = 2.5,
      p_k = 0.01,
      p_2s = 0.02,
      df = 9.0
    )
  ))

  data <- data.frame(Y = 1:10, X1 = rnorm(10), X2 = rnorm(10))
  data$X2[c(3, 7)] <- NA

  httr2::with_mocked_responses(mock_resp, {
    res <- ci_test(data, y = "Y", imputer = "null", classifier = "logistic",
                   m = 2L, n_folds = 3L)
  })

  expect_type(res, "list")
  expect_equal(res$model_id, "abc-123")
  expect_true("t_k" %in% names(res$results))
})

test_that("get_summary parses response", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(
    model_id = "abc-123",
    outcome = "Y",
    imputer = "NullImputer",
    classifier = "LogisticClassifier",
    variance_method = "mi_crossfit",
    mean_difference = 0.05,
    t_statistic = 2.5,
    df = 9,
    p_value = 0.01,
    p_value_two_sided = 0.02
  ))

  httr2::with_mocked_responses(mock_resp, {
    res <- get_summary("abc-123")
  })

  expect_equal(res$outcome, "Y")
  expect_equal(res$t_statistic, 2.5)
})

test_that("imputer_r2 parses response", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(
    model_id = "abc-123",
    mean_r2 = 0.85,
    per_variable = list(X1 = 0.9, X2 = 0.8)
  ))

  httr2::with_mocked_responses(mock_resp, {
    res <- imputer_r2("abc-123")
  })

  expect_equal(res$mean_r2, 0.85)
  expect_true("X1" %in% names(res$per_variable))
})

test_that("compute_kappa returns numeric", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(kappa = 0.042))

  httr2::with_mocked_responses(mock_resp, {
    k <- compute_kappa(0.5, 0.3, 0.2)
  })

  expect_type(k, "double")
  expect_equal(k, 0.042)
})

test_that("kappa_calibration_table returns data frame", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(
    columns = list("r2_x_z", "beta_yx", "gamma_x", "kappa", "abs_kappa"),
    data = list(
      list(0.5, 0.3, 0.2, 0.042, 0.042),
      list(0.5, 0.3, 0.3, 0.063, 0.063)
    )
  ))

  httr2::with_mocked_responses(mock_resp, {
    df <- kappa_calibration_table(r2_grid = c(0.5), beta_grid = c(0.3),
                                  gamma_grid = c(0.2, 0.3))
  })

  expect_s3_class(df, "data.frame")
  expect_equal(nrow(df), 2)
  expect_true("kappa" %in% names(df))
})

test_that("calibration_pivot returns data frame", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(
    index = list(0.5, 0.7),
    columns = list("g_x = 0.2", "g_x = 0.3"),
    data = list(
      list(0.042, 0.063),
      list(0.021, 0.032)
    )
  ))

  httr2::with_mocked_responses(mock_resp, {
    piv <- calibration_pivot(beta_yx = 0.3)
  })

  expect_s3_class(piv, "data.frame")
  expect_equal(nrow(piv), 2)
})

test_that("make_dataset sends correct request", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(
    dataset_id = "ds-789",
    n = 10,
    columns = list("Y", "X1", "X2"),
    y_name = "Y",
    expl_vars = list("X1", "X2"),
    pct_missing = 10.0
  ))

  data <- data.frame(Y = 1:10, X1 = rnorm(10), X2 = rnorm(10))

  httr2::with_mocked_responses(mock_resp, {
    res <- make_dataset(data, y = "Y")
  })

  expect_equal(res$dataset_id, "ds-789")
  expect_equal(res$n, 10)
})

test_that("simulate_data sends correct request", {
  skip_on_cran()
  fake_server_running()
  on.exit(reset_server_state())

  mock_resp <- mock_json_response(list(
    dataset_id = "ds-sim",
    n = 200,
    columns = list("Y", "X1", "X2", "X3", "X4", "X5"),
    pct_missing = 25.0
  ))

  httr2::with_mocked_responses(mock_resp, {
    res <- simulate_data("single_mar", n = 200, ci = TRUE)
  })

  expect_equal(res$dataset_id, "ds-sim")
  expect_equal(res$n, 200)
})
