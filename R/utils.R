# R/utils.R


#-------------------------------------------------------------------------------


#' Get Gaussian-Hermite quadrature points
#'
#' Calculates Gaussian-Hermite quadrature points and weights for numerical integration
#'
#' @param n_points Number of quadrature points
#' @return A list containing points and weights
#' @importFrom statmod gauss.quad
#' @export
get_gh_points <- function(n_points) {
  gh <- gauss.quad(n_points, kind = "hermite")
  list(points = gh$nodes * sqrt(2), weights = gh$weights / sqrt(pi))
}


#-------------------------------------------------------------------------------


#' Setup design matrices for model fitting
#'
#' Creates design matrices for fixed effects, between-subject variance, and within-subject variance
#'
#' @param data Data frame containing model variables
#' @param mean_formula Formula for fixed effects mean model (or NULL for intercept-only)
#' @param var_bs_formula Formula for between-subject variance model
#' @param var_ws_formula Formula for within-subject variance model
#' @return A list containing design matrices X, U, and Z
#' @importFrom stats model.matrix
#' @export
setup_matrices <- function(data, mean_formula, var_bs_formula, var_ws_formula) {
  if (!is.null(mean_formula)) {
    X <- model.matrix(mean_formula, data = data)
  } else {
    X <- matrix(1, nrow = nrow(data))  # Intercept only
  }
  
  U <- model.matrix(var_bs_formula, data = data)
  Z <- model.matrix(var_ws_formula, data = data)
  
  list(X = X, U = U, Z = Z)
}


#-------------------------------------------------------------------------------


#' Prepare model data and matrices
#'
#' A utility function to prepare all necessary data and matrices for model fitting, avoiding repeated computations
#'
#' @param data Data frame containing model variables
#' @param mean_formula Formula for fixed effects mean model
#' @param var_bs_formula Formula for between-subject variance model
#' @param var_ws_formula Formula for within-subject variance model
#' @param id_var Name of ID variable column
#' @param response_var Name of response variable column
#' @param nq Number of quadrature points
#' @return A list containing all prepared data and matrices for model fitting
#' @export
prepare_model_data <- function(data, mean_formula = NULL, var_bs_formula, var_ws_formula,
                               id_var, response_var, nq) {
  # Setup matrices
  matrices <- setup_matrices(data, mean_formula, var_bs_formula, var_ws_formula)
  
  # Get quadrature points
  gh <- get_gh_points(nq)
  
  # Convert ID to numeric factor
  id_numeric <- as.numeric(as.factor(data[[id_var]]))
  
  # Extract response vector
  y <- data[[response_var]]
  
  # Get unique IDs
  unique_ids <- unique(data[[id_var]])
  
  # Pre-compute indices for each subject
  id_indices <- split(seq_len(nrow(data)), data[[id_var]])
  
  return(list(
    matrices = matrices,
    gh = gh,
    points = gh$points,   
    weights = gh$weights, 
    y = y,
    id_numeric = id_numeric - 1,  # 0-based for C++
    unique_ids = unique_ids,
    id_indices = id_indices,
    n_fixed = ncol(matrices$X),
    n_random = ncol(matrices$U),
    n_ws = ncol(matrices$Z)
  ))
}


#-------------------------------------------------------------------------------


#' Format parameters for display
#'
#' @param params Vector of parameter values
#' @param param_names Vector of parameter names (beta_1, alpha_1, etc.)
#' @param mean_formula Formula for fixed effects mean model
#' @param var_bs_formula Formula for between-subject variance model
#' @param var_ws_formula Formula for within-subject variance model
#' @param stage stage of the model (1, 2, or 3)
#' @param model_type For stage 3, the type of model
#' @export
format_params_display <- function(params, param_names, mean_formula, var_bs_formula, 
                                  var_ws_formula, stage = 1, model_type = NULL) {
  # Helper function to print section header
  print_section_header <- function(title) {
    separator <- paste(rep("-", nchar("Regression coefficients")), collapse = "")
    cat(separator, "\n")
    cat(title, "\n")
  }
  
  # Count parameters
  n_beta <- sum(grepl("^beta_", param_names))
  n_alpha <- sum(grepl("^alpha_", param_names))
  n_tau <- sum(grepl("^tau_", param_names))
  n_sigma <- sum(grepl("^sigma_", param_names))
  
  # Get variable names from formulas - FIXED to check for intercepts
  if (!is.null(mean_formula)) {
    mean_terms <- terms(mean_formula)
    has_mean_intercept <- attr(mean_terms, "intercept") == 1
    mean_vars <- attr(mean_terms, "term.labels")
  } else {
    has_mean_intercept <- TRUE  # NULL formula means intercept only
    mean_vars <- character()
  }
  
  bs_terms <- terms(var_bs_formula)
  has_bs_intercept <- attr(bs_terms, "intercept") == 1
  bs_vars <- attr(bs_terms, "term.labels")
  
  ws_terms <- terms(var_ws_formula)
  has_ws_intercept <- attr(ws_terms, "intercept") == 1
  ws_vars <- attr(ws_terms, "term.labels")
  
  # Calculate max width for alignment
  all_vars <- character()
  if (has_mean_intercept) all_vars <- c(all_vars, "intercept")
  all_vars <- c(all_vars, mean_vars)
  if (has_bs_intercept) all_vars <- c(all_vars, "intercept")
  all_vars <- c(all_vars, bs_vars)
  if (has_ws_intercept) all_vars <- c(all_vars, "intercept")
  all_vars <- c(all_vars, ws_vars)
  
  if (stage == 3) {
    all_vars <- c(all_vars, "Lin loc", "Inter", "Quad loc", "Std Dev")
  }
  max_width <- max(nchar(all_vars)) + 1  # +1 for the colon
  
  idx <- 1
  
  # Print Regression coefficients
  print_section_header("Regression coefficients")
  
  if (has_mean_intercept) {
    cat(sprintf("  %-*s %.6f\n", max_width, "intercept:", params[idx]))
    idx <- idx + 1
  }
  
  if (length(mean_vars) > 0) {
    for (var in mean_vars) {
      cat(sprintf("  %-*s %.6f\n", max_width, paste0(var, ":"), params[idx]))
      idx <- idx + 1
    }
  }
  
  # Print BS variance parameters
  print_section_header("BS variance parameters ")
  
  if (has_bs_intercept) {
    cat(sprintf("  %-*s %.6f\n", max_width, "intercept:", params[idx]))
    idx <- idx + 1
  }
  
  if (length(bs_vars) > 0) {
    for (var in bs_vars) {
      cat(sprintf("  %-*s %.6f\n", max_width, paste0(var, ":"), params[idx]))
      idx <- idx + 1
    }
  }
  
  # Print WS variance parameters
  print_section_header("WS variance parameters ")
  
  if (has_ws_intercept) {
    cat(sprintf("  %-*s %.6f\n", max_width, "intercept:", params[idx]))
    idx <- idx + 1
  }
  
  if (length(ws_vars) > 0) {
    for (var in ws_vars) {
      cat(sprintf("  %-*s %.6f\n", max_width, paste0(var, ":"), params[idx]))
      idx <- idx + 1
    }
  }
  
  # stage 3 specific parameters
  if (stage == 3 && n_sigma > 0) {
    if (model_type == "independent") {
      print_section_header("Random scale standard deviation")
      cat(sprintf("  %-*s %.6f\n", max_width, "Std Dev:", params[idx]))
    } else if (model_type == "linear") {
      print_section_header("Random linear location (mean) effect on WS variance")
      cat(sprintf("  %-*s %.6f\n", max_width, "Lin loc:", params[idx]))
      idx <- idx + 1
      print_section_header("Random scale standard deviation")
      cat(sprintf("  %-*s %.6f\n", max_width, "Std Dev:", params[idx]))
    } else if (model_type == "interaction") {
      print_section_header("Random linear location (mean) effect on WS variance")
      cat(sprintf("  %-*s %.6f\n", max_width, "Lin loc:", params[idx]))
      idx <- idx + 1
      print_section_header("Random (location & scale) interaction effect on WS variance")
      cat(sprintf("  %-*s %.6f\n", max_width, "Inter:", params[idx]))
      idx <- idx + 1
      print_section_header("Random scale standard deviation")
      cat(sprintf("  %-*s %.6f\n", max_width, "Std Dev:", params[idx]))
    } else if (model_type == "quadratic") {
      print_section_header("Random linear and quadratic location (mean) effect on WS variance")
      cat(sprintf("  %-*s %.6f\n", max_width, "Lin loc:", params[idx]))
      idx <- idx + 1
      cat(sprintf("  %-*s %.6f\n", max_width, "Quad loc:", params[idx]))
      idx <- idx + 1
      print_section_header("Random scale standard deviation")
      cat(sprintf("  %-*s %.6f\n", max_width, "Std Dev:", params[idx]))
    }
  }
}


#-------------------------------------------------------------------------------


#' Create descriptive parameter names for results table
#'
#' @param param_names Vector of parameter names (beta_1, alpha_1, etc.)
#' @param mean_formula Formula for fixed effects mean model
#' @param var_bs_formula Formula for between-subject variance model
#' @param var_ws_formula Formula for within-subject variance model
#' @param stage stage of the model (1, 2, or 3)
#' @param model_type For stage 3, the type of model
#' @return Vector of descriptive parameter names
#' @export
create_param_names_table <- function(param_names, mean_formula, var_bs_formula, 
                                     var_ws_formula, stage = 1, model_type = NULL) {
  descriptive_names <- character()
  
  # Get variable names from formulas
  if (!is.null(mean_formula)) {
    mean_terms <- terms(mean_formula)
    mean_vars <- attr(mean_terms, "term.labels")
    
    # Only add intercept if it exists in the formula
    if (attr(mean_terms, "intercept") == 1) {
      descriptive_names <- c(descriptive_names, "intercept")
    }
    if (length(mean_vars) > 0) {
      descriptive_names <- c(descriptive_names, mean_vars)
    }
  } else {
    # NULL formula means intercept only
    descriptive_names <- c(descriptive_names, "intercept")
  }
  
  # BS variance parameters
  bs_terms <- terms(var_bs_formula)
  bs_vars <- attr(bs_terms, "term.labels")
  
  # Only add BS intercept if it exists in the formula
  if (attr(bs_terms, "intercept") == 1) {
    descriptive_names <- c(descriptive_names, "BS_intercept")
  }
  if (length(bs_vars) > 0) {
    descriptive_names <- c(descriptive_names, paste0("BS_", bs_vars))
  }
  
  # WS variance parameters
  ws_terms <- terms(var_ws_formula)
  ws_vars <- attr(ws_terms, "term.labels")
  
  # Only add WS intercept if it exists in the formula
  if (attr(ws_terms, "intercept") == 1) {
    descriptive_names <- c(descriptive_names, "WS_intercept")
  }
  if (length(ws_vars) > 0) {
    descriptive_names <- c(descriptive_names, paste0("WS_", ws_vars))
  }
  
  # stage 3 specific parameters (unchanged)
  if (stage == 3) {
    n_sigma <- sum(grepl("^sigma_", param_names))
    if (n_sigma > 0) {
      if (model_type == "independent") {
        descriptive_names <- c(descriptive_names, "Std_Dev")
      } else if (model_type == "linear") {
        descriptive_names <- c(descriptive_names, "Lin_loc")
        descriptive_names <- c(descriptive_names, "Std_Dev")
      } else if (model_type == "interaction") {
        descriptive_names <- c(descriptive_names, "Lin_loc")
        descriptive_names <- c(descriptive_names, "Inter")
        descriptive_names <- c(descriptive_names, "Std_Dev")
      } else if (model_type == "quadratic") {
        descriptive_names <- c(descriptive_names, "Lin_loc")
        descriptive_names <- c(descriptive_names, "Quad_loc")
        descriptive_names <- c(descriptive_names, "Std_Dev")
      }
    }
  }
  
  return(descriptive_names)
}


#-------------------------------------------------------------------------------


#' Update Empirical Bayes estimates for stage 1 & 2
#'
#' This function calls C++ function to calculate the EB estimates.
#'
#' @param params Vector of current model parameters (beta, alpha, tau).
#' @param model_data List containing model matrices and data.
#' @param eb_prev_theta EB mean estimates from the previous iteration.
#' @param eb_prev_psd EB standard deviation estimates from the previous iteration.
#' @return A list containing the newly calculated `theta_eb` and `psd_eb`.
#' @export
update_eb_estimates <- function(params, model_data, eb_prev_theta, eb_prev_psd) {
  return(update_eb_estimates_rcpp(
    params = params,
    X = model_data$matrices$X,
    U = model_data$matrices$U,
    Z = model_data$matrices$Z,
    y = model_data$y,
    id = model_data$id_numeric,
    points = model_data$gh$points,
    weights = model_data$gh$weights,
    eb_prev_theta = eb_prev_theta,
    eb_prev_psd = eb_prev_psd
  ))
}

#' Update Bivariate Empirical Bayes estimates for stage 3
#'
#' @param params Vector of current model parameters.
#' @param model_data List containing model matrices and data.
#' @param model_type The specific stage 3 model type.
#' @param eb_prev_theta2 A matrix of the posterior means from the previous iteration.
#' @param eb_prev_thetav A matrix of the posterior VCV elements from the previous iteration.
#' @param adaptive An integer (0 or 1) indicating whether to use adaptive quadrature.
#' @return A list containing the newly calculated `theta2_eb` and `thetav_eb`.
#' @export
update_eb_estimates_stage3 <- function(params, model_data, model_type,
                                       eb_prev_theta2, eb_prev_thetav, adaptive) {
  return(update_eb_estimates_stage3_rcpp(
    params = params,
    X = model_data$matrices$X,
    U = model_data$matrices$U,
    Z = model_data$matrices$Z,
    y = model_data$y,
    id = model_data$id_numeric,
    points = model_data$gh$points,
    weights = model_data$gh$weights,
    model_type = model_type,
    adaptive = adaptive,
    eb_prev_theta2 = eb_prev_theta2,
    eb_prev_thetav = eb_prev_thetav
  ))
}


#-------------------------------------------------------------------------------


#' Compute standardized residuals for stage 1 & 2
#'
#' Computes standardized residuals using empirical Bayes estimates.
#'
#' @param params Vector of model parameters (beta, alpha, tau)
#' @param model_data List containing model matrices and data
#' @param eb_estimates List containing empirical Bayes estimates
#' @return Vector of standardized residuals
#' @export
compute_standardized_residuals <- function(params, model_data, eb_estimates) {
  if (is.null(eb_estimates)) {
    return(NULL)
  }
  
  # Extract parameters
  n_fixed <- model_data$n_fixed
  n_random <- model_data$n_random
  beta <- params[1:n_fixed]
  alpha <- params[(n_fixed + 1):(n_fixed + n_random)]
  tau <- params[(n_fixed + n_random + 1):length(params)]
  
  # Get data
  X <- model_data$matrices$X
  U <- model_data$matrices$U
  Z <- model_data$matrices$Z
  y <- model_data$y
  id <- model_data$id_numeric + 1 # Convert from 0-based C++ index to 1-based R index
  
  # Calculate all components in vectorized form
  XB <- as.vector(X %*% beta)
  BS_var <- exp(as.vector(U %*% alpha))
  WS_var <- exp(as.vector(Z %*% tau))
  
  # Expand the per-subject EB estimates to match the full data length
  theta_eb_long <- eb_estimates$theta_eb[id]
  
  # Compute predicted values and residuals in a single operation
  predicted <- XB + sqrt(BS_var) * theta_eb_long
  std_residuals <- (y - predicted) / sqrt(WS_var)
  
  return(std_residuals)
}


#-------------------------------------------------------------------------------


#' Compute standardized residuals for stage 3
#'
#' Computes standardized residuals using the bivariate empirical Bayes estimates 
#' from stage 3.
#'
#' @param params Vector of final model parameters from stage 3.
#' @param model_data List containing model matrices and data.
#' @param model_type The specific stage 3 model type.
#' @param eb_estimates List containing final bivariate EB estimates.
#' @return A vector of standardized residuals.
#' @export
compute_standardized_residuals_stage3 <- function(params, model_data, model_type, eb_estimates) {
  if (is.null(eb_estimates)) { return(NULL) }
  
  # Extract parameters
  n_fixed <- model_data$n_fixed
  n_random_bs <- model_data$n_random
  n_random_ws <- model_data$n_ws
  
  beta <- params[1:n_fixed]
  alpha <- params[(n_fixed + 1):(n_fixed + n_random_bs)]
  tau <- params[(n_fixed + n_random_bs + 1):(n_fixed + n_random_bs + n_random_ws)]
  
  n_spar <- switch(model_type, "independent" = 1, "linear" = 2, "quadratic" = 3, "interaction" = 3)
  spar_start_idx <- length(params) - n_spar + 1
  spar <- params[spar_start_idx:length(params)]
  
  # Extract data
  X <- model_data$matrices$X; U <- model_data$matrices$U; Z <- model_data$matrices$Z
  y <- model_data$y; id <- model_data$id_numeric + 1 # To 1-based R index
  
  # Expand per-subject EB estimates to match full data length
  theta1_eb_long <- eb_estimates$theta2_eb[id, 1]
  theta2_eb_long <- eb_estimates$theta2_eb[id, 2]
  
  # Calculate components in vectorized form
  XB <- as.vector(X %*% beta)
  BS_var <- exp(as.vector(U %*% alpha))
  Z_tau_obs <- as.vector(Z %*% tau)
  
  # Predicted mean
  predicted_mean <- XB + sqrt(BS_var) * theta1_eb_long
  
  # Predicted WS variance
  log_wsvar <- switch(model_type,
                      "independent" = Z_tau_obs + spar[1] * theta2_eb_long,
                      "linear"      = Z_tau_obs + spar[1] * theta1_eb_long + spar[2] * theta2_eb_long,
                      "quadratic"   = Z_tau_obs + spar[1] * theta1_eb_long + spar[2] * theta1_eb_long^2 + spar[3] * theta2_eb_long,
                      "interaction" = Z_tau_obs + spar[1] * theta1_eb_long + spar[2] * theta1_eb_long * theta2_eb_long + spar[3] * theta2_eb_long
  )
  predicted_ws_var <- exp(log_wsvar)
  
  # Compute standardized residuals in a single operation
  std_residuals <- (y - predicted_mean) / sqrt(predicted_ws_var)
  
  return(std_residuals)
}