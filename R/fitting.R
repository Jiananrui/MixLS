# R/fitting.R


#-------------------------------------------------------------------------------


#' Fit stage 1 of Mixed Effects Location Scale Model
#'
#' stage 1 always uses ~ 1 for within-subject variance to provide starting values
#' for subsequent stages. This ensures consistent initialization regardless of the
#' final model specification.
#'
#' @param data Data frame containing variables
#' @param var_bs_formula Formula for between-subject variance
#' @param mean_formula Formula for fixed effects mean (NULL for intercept only)
#' @param var_ws_formula Formula for within-subject variance (should always be ~ 1 for stage 1)
#' @param id_var Name of ID variable
#' @param response_var Name of response variable
#' @param nq Number of quadrature points
#' @param maxiter Maximum iterations
#' @param tol Convergence tolerance
#' @param init_ridge Initial ridge parameter value (default: 0)
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @return List of model results
#' @importFrom stats pnorm terms
#' @export

stage1_fit <- function(data, var_bs_formula, mean_formula = NULL, var_ws_formula,
                       id_var, response_var, nq, maxiter, tol, init_ridge = 0, adaptive = 1) {
  # Prepare model data
  model_data <- prepare_model_data(
    data, mean_formula, var_bs_formula, var_ws_formula, id_var, response_var, nq
  )

  # Initialize parameters with EM refinement
  init_results <- init_1_with_em(
    data, model_data$matrices, mean_formula, var_bs_formula, var_ws_formula, id_var, response_var
  )
  params <- init_results$params

  param_names <- c(
    paste0("beta_", 1:model_data$n_fixed), paste0("alpha_", 1:model_data$n_random), paste0("tau_", 1:model_data$n_ws)
  )

  # Initialization for the loop
  ridge <- init_ridge
  ridge_iter_count <- 0
  converged <- FALSE
  eb_prev_theta <- NULL
  eb_prev_psd <- NULL

  if (adaptive == 1) {
    alpha1_init <- params[model_data$n_fixed + 1]
    scaling_factor <- sqrt(exp(alpha1_init))

    eb_prev_theta <- init_results$theta_em / scaling_factor
    eb_prev_psd <- sqrt(pmax(init_results$pv_em, 1e-8)) / scaling_factor

    cat("\nAdaptive quadrature enabled. Initial EB estimates scaled.\n")
  } else {
    cat("\nUsing FIXED quadrature.\n")
  }

  cat("\nStarting optimization:\nInitial parameters:\n")
  format_params_display(params, param_names, mean_formula, var_bs_formula,
                        var_ws_formula, stage = 1)

  prev_loglik <- stage12_loglik_wrapper(params, model_data, adaptive, eb_prev_theta, eb_prev_psd)
  cat("Initial log-likelihood:", prev_loglik, "\n")

  # Main optimization loop
  for (iter in 1:maxiter) {
    # Gradient & Hessian calculation
    grad <- stage12_gradient_wrapper(params, model_data, adaptive, eb_prev_theta, eb_prev_psd)
    hess <- stage12_hessian_wrapper(params, model_data, adaptive, eb_prev_theta, eb_prev_psd)

    # Parameter update setup
    hess_adj <- hess
    diag(hess_adj) <- diag(hess_adj) * (1 + ridge)
    delta <- try(solve_matrix_rcpp(hess_adj, grad), silent = TRUE)

    if (inherits(delta, "try-error") || length(delta) == 0) {
      ridge <- ridge + 0.1
      ridge_iter_count <- 0
      cat(sprintf("\nIteration %d: Hessian inversion failed. Increasing ridge to %.3f\n", iter, ridge))
      next
    }

    new_params <- params - delta

    # Log-likelihood check
    current_loglik <- stage12_loglik_wrapper(new_params, model_data, adaptive, eb_prev_theta, eb_prev_psd)

    # Print full iteration details
    loglik_change <- current_loglik - prev_loglik
    cat("\n")
    separator2 <- paste(rep("=", nchar("WS variance parameters ")), collapse = "")
    cat(separator2)
    cat(sprintf("\nIteration %d \n", iter))
    cat("Updated parameters:\n")
    format_params_display(new_params, param_names, mean_formula, var_bs_formula, var_ws_formula, stage = 1)
    cat(sprintf("Log-likelihood: %.6f | Change: %.5f | Ridge: %.3f\n",
                current_loglik, current_loglik - prev_loglik, ridge))

    # Check improvement
    rel_change <- loglik_change / (abs(prev_loglik))
    if (!is.finite(current_loglik) || (iter > 1 && rel_change < -0.000001)) {

      ridge  <- ridge + 0.1
      ridge_iter_count <- 0
      cat("Warning: Non-finite or decreasing log-likelihood. Increasing ridge.\n")
      next
    }

    # Convergence check
    if (max(abs(grad)) < tol) {
      cat(paste0("\nConverged at iteration: ", iter, "\n"))
      converged <- TRUE
      params <- new_params
      prev_loglik <- current_loglik
      break
    }

    # Accept the step
    params <- new_params
    prev_loglik <- current_loglik

    # Update EB estimates for the next iteration
    if (adaptive == 1) {
      eb_next <- update_eb_estimates(params, model_data, eb_prev_theta, eb_prev_psd)
      eb_prev_theta <- eb_next$theta_eb
      eb_prev_psd <- eb_next$psd_eb
    }

    ridge_iter_count <- ridge_iter_count + 1
    if (ridge_iter_count >= 10 && ridge > 0) {
      ridge <- 0
      ridge_iter_count <- 0
      cat("Resetting ridge to 0 after 10 stable iterations\n")
    }
  }

  if (!converged) {
    warning("Maximum iterations reached without full convergence.")
  }

  # Final computations post-loop
  hess_final <- stage12_hessian_wrapper(params, model_data, adaptive, theta_eb = eb_prev_theta, psd_eb = eb_prev_psd)
  inv_hess <- try(solve(-hess_final), silent = TRUE)

  se <- if (inherits(inv_hess, "try-error")) {
    warning("Failed to invert Hessian for standard errors")
    rep(NA, length(params))
  } else {
    sqrt(pmax(diag(inv_hess), 0))
  }

  z_scores <- params / se
  p_values <- 2 * (1 - pnorm(abs(z_scores)))
  lower_ci <- params - 1.96 * se
  upper_ci <- params + 1.96 * se

  descriptive_names <- create_param_names_table(param_names, mean_formula, var_bs_formula,
                                                var_ws_formula, stage = 1)

  results_table <- data.frame(
    Parameter = descriptive_names, Estimate = params, SE = se,
    `Z-score` = z_scores, `P-value` = p_values,
    `Lower-CI` = lower_ci, `Upper-CI` = upper_ci,
    stringsAsFactors = FALSE
  )
  names(results_table) <- c("Parameter", "Estimate", "SE", "Z-score", "P-value", "Lower-CI", "Upper-CI")

  n_params <- length(params)
  n_subjects <- length(model_data$unique_ids)
  aic <- -2 * prev_loglik + 2 * n_params
  bic <- -2 * prev_loglik + log(n_subjects) * n_params

  final_eb_estimates <- NULL
  standardized_residuals <- NULL

  if (adaptive == 1) {
    final_eb_estimates <- update_eb_estimates(params, model_data, eb_prev_theta, eb_prev_psd)
    standardized_residuals <- compute_standardized_residuals(params, model_data, final_eb_estimates)
  }

  return(list(
    results_table = results_table,
    final_loglik = prev_loglik,
    AIC = aic,
    BIC = bic,
    convergence = ifelse(converged, "Converged", "Max iterations reached"),
    iterations = ifelse(converged, iter, maxiter),
    final_ridge = ridge,
    init_results = init_results,
    model_data = model_data,
    adaptive_used = adaptive,
    eb_estimates = final_eb_estimates,
    standardized_residuals = standardized_residuals
  ))
}


#-------------------------------------------------------------------------------


#' Fit stage 2 of Mixed Effects Location Scale Model
#'
#' @param stage1_result Results from stage 1 fitting
#' @param data Data frame containing variables
#' @param var_bs_formula Formula for between-subject variance
#' @param mean_formula Formula for fixed effects mean (NULL for intercept only)
#' @param var_ws_formula Formula for within-subject variance
#' @param id_var Name of ID variable
#' @param response_var Name of response variable
#' @param nq Number of quadrature points
#' @param maxiter Maximum iterations
#' @param tol Convergence tolerance
#' @param init_ridge Initial ridge parameter value (default: 0.1)
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @return List of model results
#' @importFrom stats pnorm terms
#' @export
stage2_fit <- function(stage1_result, data, var_bs_formula, mean_formula, var_ws_formula,
                       id_var, response_var, nq, maxiter, tol, init_ridge = 0.1, adaptive = 1) {

  # Prepare model data for stage 2
  model_data <- prepare_model_data(
    data, mean_formula, var_bs_formula, var_ws_formula,
    id_var, response_var, nq
  )

  # Get the WS_intercept value from stage 1 results
  ws_intercept_idx <- grep("^WS_intercept$", stage1_result$results_table$Parameter)[1]
  tau_intercept_value <- stage1_result$results_table$Estimate[ws_intercept_idx]

  # Initialize stage 2 parameters
  init_results2 <- init_2(
    data = data,
    response_var = response_var,
    mean_fit = stage1_result$init_results$mean_fit,
    tau_intercept = tau_intercept_value,
    var_ws_formula = var_ws_formula
  )

  # Get stage 1 parameters and descriptive names
  stage1_estimates <- stage1_result$results_table$Estimate
  stage1_descriptive_names <- stage1_result$results_table$Parameter

  # Count how many parameters of each type we should have from formulas
  if (!is.null(mean_formula)) {
    mean_terms <- terms(mean_formula)
    mean_vars <- attr(mean_terms, "term.labels")
    n_beta_expected <- length(mean_vars) + attr(mean_terms, "intercept")
  } else {
    n_beta_expected <- 1  # NULL formula means intercept only
  }

  bs_terms <- terms(var_bs_formula)
  bs_vars <- attr(bs_terms, "term.labels")
  n_alpha_expected <- length(bs_vars) + attr(bs_terms, "intercept")

  # Extract beta parameters (regression coefficients)
  beta_params <- stage1_estimates[1:n_beta_expected]

  # Extract alpha parameters (BS variance parameters)
  alpha_start <- n_beta_expected + 1
  alpha_end <- alpha_start + n_alpha_expected - 1
  alpha_params <- stage1_estimates[alpha_start:alpha_end]

  # Combine parameters: beta and alpha from stage 1, new tau from init_2
  params <- c(
    beta_params,     # beta parameters
    alpha_params,    # alpha parameters
    init_results2    # new tau parameters
  )

  param_names <- c(
    paste0("beta_", 1:model_data$n_fixed),
    paste0("alpha_", 1:model_data$n_random),
    paste0("tau_", 1:model_data$n_ws)
  )

  # Verify lengths match
  if (length(params) != length(param_names)) {
    stop(sprintf("Parameter length mismatch: %d parameters but %d names\nn_fixed: %d, n_random: %d, n_ws: %d\nbeta params: %d, alpha params: %d, tau params: %d",
                 length(params), length(param_names),
                 model_data$n_fixed, model_data$n_random, model_data$n_ws,
                 length(beta_params), length(alpha_params), length(init_results2)))
  }

  prev_loglik <- -Inf
  ridge <- init_ridge
  ridge_iter_count <- 0
  converged <- FALSE
  eb_prev_theta <- NULL
  eb_prev_psd <- NULL

  # Initialize EB estimates for adaptive quadrature
  if (adaptive == 1) {
    if (is.null(stage1_result$eb_estimates)) {
      stop("Adaptive quadrature for stage 2 requires EB estimates from stage 1. Please run stage 1 with adaptive=1.")
    }
    eb_prev_theta <- stage1_result$eb_estimates$theta_eb
    eb_prev_psd <- stage1_result$eb_estimates$psd_eb
    cat("\nAdaptive quadrature enabled. Using final EB estimates from stage 1.\n")
  } else {
    cat("\nUsing FIXED quadrature.\n")
  }

  cat("\nStarting stage 2 optimization:\nInitial parameters:\n")
  format_params_display(params, param_names, mean_formula, var_bs_formula,
                        var_ws_formula, stage = 2)


  prev_loglik <- stage12_loglik_wrapper(params, model_data, adaptive, eb_prev_theta, eb_prev_psd)
  cat("Initial log-likelihood:", prev_loglik, "\n")

  # Main optimization loop
  for (iter in 1:maxiter) {
    # Gradient & Hessian calculation
    grad <- stage12_gradient_wrapper(params, model_data, adaptive, eb_prev_theta, eb_prev_psd)
    hess <- stage12_hessian_wrapper(params, model_data, adaptive, eb_prev_theta, eb_prev_psd)
    hess_adj <- hess

    # stage specific ridge management
    if (iter == 5) {
      ridge <- init_ridge # Reset to the initial stage 2 ridge
      ridge_iter_count <- 0
      cat("Resetting ridge to initial stage 2 value at iteration 5.\n")
    } else {
      ridge_iter_count <- ridge_iter_count + 1
      if (ridge_iter_count >= 10 && ridge > 0) {
        ridge <- 0
        ridge_iter_count <- 0
        cat("Resetting ridge to 0 after 10 stable iterations.\n")
      }
    }

    diag(hess_adj) <- diag(hess_adj) * (1 + ridge)
    delta <- try(solve_matrix_rcpp(hess_adj, grad), silent = TRUE)

    if (inherits(delta, "try-error") || length(delta) == 0) {
      ridge <- ridge + 0.1
      ridge_iter_count <- 0
      cat(sprintf("\nIteration %d: Hessian inversion failed. Increasing ridge to %.3f\n", iter, ridge))
      next
    }

    # Step halving if NS>0 and iter <=5
    if (iter <= 5) {
      delta <- 0.5 * delta
    }

    new_params <- params - delta
    current_loglik <- stage12_loglik_wrapper(new_params, model_data, adaptive, eb_prev_theta, eb_prev_psd)

    # Print iteration information
    loglik_change <- current_loglik - prev_loglik
    cat("\n")
    separator2 <- paste(rep("=", nchar("WS variance parameters ")), collapse = "")
    cat(separator2)
    cat(sprintf("\nIteration %d \n", iter))
    cat("Updated parameters:\n")
    format_params_display(new_params, param_names, mean_formula, var_bs_formula, var_ws_formula, stage = 2)
    cat(sprintf("Log-likelihood: %.6f | Change: %.5f | Ridge: %.3f\n",
                current_loglik, current_loglik - prev_loglik, ridge))

    # Check improvement
    rel_change <- loglik_change / (abs(prev_loglik))
    if (!is.finite(current_loglik) || (iter > 1 && rel_change < -0.000001)) {
      ridge <- ridge + 0.1
      ridge_iter_count <- 0
      cat("Warning: Non-finite or decreasing log-likelihood. Increasing ridge.\n")
      next
    }

    # Convergence check
    if (max(abs(grad)) < tol) {
      cat(paste0("\nConverged at iteration: ", iter, "\n"))
      converged <- TRUE
      params <- new_params
      prev_loglik <- current_loglik
      break
    }

    # Accept step and prepare for next iteration
    params <- new_params
    prev_loglik <- current_loglik

    if (adaptive == 1) {
      eb_next <- update_eb_estimates(params, model_data, eb_prev_theta, eb_prev_psd)
      eb_prev_theta <- eb_next$theta_eb
      eb_prev_psd <- eb_next$psd_eb
    }
  }

  if (!converged) {
    warning("Maximum iterations reached without full convergence.")
  }

  # Final calculations
  hess_final <- stage12_hessian_wrapper(params, model_data, adaptive, theta_eb = eb_prev_theta, psd_eb = eb_prev_psd)
  inv_hess <- try(solve(-hess_final), silent = TRUE)

  se <- if (inherits(inv_hess, "try-error")) {
    warning("Failed to invert Hessian for standard errors")
    rep(NA, length(params))
  } else {
    sqrt(pmax(diag(inv_hess), 0))
  }

  z_scores <- params / se
  p_values <- 2 * (1 - pnorm(abs(z_scores)))
  lower_ci <- params - 1.96 * se
  upper_ci <- params + 1.96 * se

  descriptive_names <- create_param_names_table(param_names, mean_formula, var_bs_formula,
                                                var_ws_formula, stage = 2)

  results_table <- data.frame(
    Parameter = descriptive_names, Estimate = params, SE = se,
    `Z-score` = z_scores, `P-value` = p_values,
    `Lower-CI` = lower_ci, `Upper-CI` = upper_ci,
    stringsAsFactors = FALSE
  )
  names(results_table) <- c("Parameter", "Estimate", "SE", "Z-score", "P-value", "Lower-CI", "Upper-CI")

  n_params <- length(params)
  n_subjects <- length(model_data$unique_ids)
  aic <- -2 * prev_loglik + 2 * n_params
  bic <- -2 * prev_loglik + log(n_subjects) * n_params

  final_eb_estimates <- NULL
  standardized_residuals <- NULL

  if (adaptive == 1) {
    final_eb_estimates <- update_eb_estimates(params, model_data, eb_prev_theta, eb_prev_psd)
    standardized_residuals <- compute_standardized_residuals(params, model_data, final_eb_estimates)
  }

  return(list(
    results_table = results_table,
    final_loglik = prev_loglik,
    AIC = aic,
    BIC = bic,
    convergence = ifelse(converged, "Converged", "Max iterations reached"),
    iterations = ifelse(converged, iter, maxiter),
    final_ridge = ridge,
    init_results = init_results2,
    model_data = model_data,
    adaptive_used = adaptive,
    eb_estimates = final_eb_estimates,
    standardized_residuals = standardized_residuals
  ))
}


#-------------------------------------------------------------------------------


#' Fit stage 3 of Mixed Effects Location Scale Model
#'
#' @param stage2_result Results from stage 2 fitting (or stage 1 if stage 2 is skipped)
#' @param data Data frame containing variables
#' @param mean_formula Formula for fixed effects mean
#' @param var_bs_formula Formula for between-subject variance
#' @param var_ws_formula Formula for within-subject variance
#' @param id_var Name of ID variable
#' @param response_var Name of response variable
#' @param nq Number of quadrature points
#' @param maxiter Maximum iterations
#' @param tol Convergence tolerance
#' @param model_type Type of model: "independent", "linear", "interaction", or "quadratic"
#' @param init_ridge Initial ridge parameter value (default: 0.2)
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @return List of model results
#' @export
stage3_fit <- function(stage2_result, data, mean_formula, var_bs_formula, var_ws_formula,
                       id_var, response_var, nq, maxiter, tol,
                       model_type, init_ridge = 0.2, adaptive = 1) {
  # Prepare model data
  model_data <- prepare_model_data(
    data, mean_formula, var_bs_formula, var_ws_formula,
    id_var, response_var, nq
  )

  # Parameter initialization
  params <- init_3(
    data = data, id_var = id_var, response_var = response_var,
    fit2 = stage2_result, model_type = model_type
  )

  n_fixed <- ncol(model_data$matrices$X)
  n_random <- ncol(model_data$matrices$U)
  n_bs <- ncol(model_data$matrices$Z)
  n_ws <- switch(model_type, "independent" = 1, "linear" = 2, "interaction" = 3, "quadratic" = 3)

  param_names <- c(
    paste0("beta_", 1:n_fixed), paste0("alpha_", 1:n_random), paste0("tau_", 1:n_bs),
    paste0("sigma_", 1:n_ws)
  )

  # Bivariate EB estimate initialization
  eb_prev_theta2 <- NULL
  eb_prev_thetav <- NULL
  n_subjects <- length(model_data$unique_ids)

  if (adaptive == 1) {
    if (is.null(stage2_result$eb_estimates)) {
      warning("Adaptive quadrature for stage 3 requires EB estimates from previous stage. Falling back to fixed quadrature.")
      adaptive <- 0
    } else {
      # Initialize bivariate EB structures
      eb_prev_theta2 <- matrix(0.0, nrow = n_subjects, ncol = 2)
      eb_prev_thetav <- matrix(0.0, nrow = n_subjects, ncol = 3)

      # Use stage 2 EB estimates for the first random effect (location)
      eb_prev_theta2[, 1] <- stage2_result$eb_estimates$theta_eb
      eb_prev_thetav[, 1] <- stage2_result$eb_estimates$psd_eb^2

      # Initialize second random effect (scale) posterior variance to 1.0, mean & cov to 0.0
      eb_prev_thetav[, 3] <- 1.0

      cat("\nAdaptive quadrature enabled for stage 3. Using final EB estimates from previous stage to initialize.\n")
    }
  } else {
    cat("\nUsing FIXED quadrature for stage 3.\n")
  }

  # Optimization loop
  prev_loglik <- -Inf
  ridge <- init_ridge
  stable_count <- 0
  converged <- FALSE

  cat(paste0("\nStarting stage 3 optimization (", model_type, " model):\n"))
  cat("Initial parameters:\n")
  format_params_display(params, param_names, mean_formula, var_bs_formula,
                        var_ws_formula, stage = 3, model_type = model_type)

  # Calculate initial log-likelihood using the correct wrapper and EB estimates
  prev_loglik <- stage3_loglik_wrapper(params, model_data, model_type, adaptive,
                                       eb_theta2 = eb_prev_theta2, eb_thetav = eb_prev_thetav)
  cat("Initial log-likelihood:", prev_loglik, "\n")

  # Main optimization loop
  for (iter in 1:maxiter) {
    if (iter == 10) {
      ridge <- 0
      stable_count <- 0
    }

    # Gradient and Hessian calculation
    grad <- stage3_gradient_wrapper(params, model_data, model_type, adaptive,
                                    eb_theta2 = eb_prev_theta2, eb_thetav = eb_prev_thetav)
    hess <- stage3_hessian_wrapper(params, model_data, model_type, adaptive,
                                   eb_theta2 = eb_prev_theta2, eb_thetav = eb_prev_thetav)

    # Parameter update
    hess_adj <- hess
    diag(hess_adj) <- diag(hess_adj) * (1 + ridge)
    delta <- try(solve_matrix_rcpp(hess_adj, grad), silent = TRUE)

    if (inherits(delta, "try-error") || length(delta) == 0) {
      ridge <- ridge + 0.1
      cat(sprintf("\nIteration %d: Hessian inversion failed. Increasing ridge to %.3f\n", iter, ridge))
      next
    }

    # Step halving
    if (iter <= 5) { delta <- 0.5 * delta }
    new_params <- params - delta

    # Check log-likelihood
    current_loglik <- stage3_loglik_wrapper(new_params, model_data, model_type, adaptive,
                                            eb_theta2 = eb_prev_theta2, eb_thetav = eb_prev_thetav)

    # Print iteration information
    cat("\n"); cat(paste(rep("=", 40), collapse="")); cat("\n")
    cat(sprintf("Iteration %d\n", iter))
    format_params_display(new_params, param_names, mean_formula, var_bs_formula,
                          var_ws_formula, stage = 3, model_type = model_type)
    cat(sprintf("Log-likelihood: %.6f | Change: %.5f | Ridge: %.3f\n",
                current_loglik, current_loglik - prev_loglik, ridge))

    rel_change <- (current_loglik - prev_loglik) / (abs(prev_loglik))

    if (!is.finite(current_loglik) || (iter > 1 && rel_change < -0.000001)) {
      ridge <- ridge + 0.1
      stable_count <- 0
      cat("Warning: Non-finite or decreasing log-likelihood. Increasing ridge.\n")
      next
    }

    if (max(abs(grad)) < tol) {
      cat(paste0("Converged at iteration: ", iter, "\n"))
      converged <- TRUE
      params <- new_params
      prev_loglik <- current_loglik
      break
    }

    # Accept step and update for next iteration
    params <- new_params
    prev_loglik <- current_loglik

    # Update bivariate EB estimates for next iteration
    if (adaptive == 1) {
      eb_next <- update_eb_estimates_stage3(params, model_data, model_type,
                                            eb_prev_theta2, eb_prev_thetav, adaptive)
      eb_prev_theta2 <- eb_next$theta2_eb
      eb_prev_thetav <- eb_next$thetav_eb
    }

    stable_count <- stable_count + 1
    if (stable_count >= 10 && ridge > 0) {
      ridge <- 0
      stable_count <- 0
      cat("Resetting ridge to 0 after 10 stable iterations\n")
    }
  }

  if (!converged) { warning("Maximum iterations reached without full convergence.") }

  # Final computations
  hess_final <- stage3_hessian_wrapper(params, model_data, model_type, adaptive,
                                       eb_theta2 = eb_prev_theta2, eb_thetav = eb_prev_thetav)
  inv_hess <- try(solve(-hess_final), silent = TRUE)

  se <- if (inherits(inv_hess, "try-error")) {
    warning("Failed to invert Hessian for standard errors")
    rep(NA, length(params))
  } else {
    sqrt(pmax(diag(inv_hess), 0))
  }

  # Final EB estimates & residuals
  final_eb_estimates <- NULL
  standardized_residuals <- NULL

  if (adaptive == 1) {
    # Final update of EB estimates with converged parameters
    final_eb_estimates <- update_eb_estimates_stage3(params, model_data, model_type,
                                                     eb_prev_theta2, eb_prev_thetav, adaptive)
    # Compute residuals using the final bivariate estimates
    standardized_residuals <- compute_standardized_residuals_stage3(params, model_data, model_type, final_eb_estimates)
  }

  z_scores <- params / se
  p_values <- 2 * (1 - pnorm(abs(z_scores)))
  lower_ci <- params - 1.96 * se
  upper_ci <- params + 1.96 * se
  descriptive_names <- create_param_names_table(param_names, mean_formula, var_bs_formula,
                                                var_ws_formula, stage = 3, model_type = model_type)
  results_table <- data.frame(
    Parameter = descriptive_names, Estimate = params, SE = se,
    `Z-score` = z_scores, `P-value` = p_values,
    `Lower-CI` = lower_ci, `Upper-CI` = upper_ci
  )
  names(results_table) <- c("Parameter", "Estimate", "SE", "Z-score", "P-value", "Lower-CI", "Upper-CI")

  aic <- -2 * prev_loglik + 2 * length(params)
  bic <- -2 * prev_loglik + log(n_subjects) * length(params)

  return(list(
    results_table = results_table,
    final_loglik = prev_loglik,
    AIC = aic,
    BIC = bic,
    convergence = ifelse(converged, "Converged", "Max iterations reached"),
    iterations = ifelse(converged, iter, maxiter),
    final_ridge = ridge,
    model_type = model_type,
    model_data = model_data,
    adaptive_used = adaptive,
    eb_estimates = final_eb_estimates,
    standardized_residuals = standardized_residuals
  ))
}

