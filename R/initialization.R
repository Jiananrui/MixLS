# R/initialization.R


#-------------------------------------------------------------------------------


#' Initialize stage 1 parameters
#'
#' Computes initial values for Stage 1 model parameters based on OLS fits
#'
#' @param data Data frame containing model variables
#' @param mean_formula Formula for fixed effects mean model
#' @param var_bs_formula Formula for between-subject variance model (or NULL)
#' @param var_ws_formula Formula for within-subject variance model
#' @param response_var Name of response variable column
#' @return List of initial parameter values and supporting information
init_1 <- function(data, mean_formula, 
                   var_bs_formula = NULL, var_ws_formula = NULL, response_var) {
  # Fit OLS for the mean model (beta)
  ols_fit <- lm(as.formula(paste(response_var, paste(deparse(mean_formula), collapse = " "), sep = " ")), data = data)
  beta_init <- coef(ols_fit)
  
  # Residuals and errv
  r <- residuals(ols_fit)
  errv <- mean(r^2)
  log_r2 <- log(r^2 + .Machine$double.eps)
  
  # tau initialization
  if (!is.null(var_ws_formula)) {
    ws_data <- data
    ws_data$log_r2 <- log_r2
    # Fit ws model and then override the intercept if it exists
    ws_fit <- lm(update(var_ws_formula, log_r2 ~ .), data = ws_data)
    tau <- coef(ws_fit)
    
    # Only override intercept if it exists in the formula
    ws_terms <- terms(var_ws_formula)
    if (attr(ws_terms, "intercept") == 1) {
      tau[1] <- log(0.6 * errv)  # Override intercept
    }
  } else {
    tau <- log(0.6 * errv)
    names(tau) <- "tau_intercept"
  }
  
  # alpha initialization
  if (!is.null(var_bs_formula)) {
    bs_data <- data
    bs_data$log_r2 <- log_r2
    # Fit bs model and then override the intercept if it exists
    bs_fit <- lm(update(var_bs_formula, log_r2 ~ .), data = bs_data)
    alpha <- coef(bs_fit)
    
    # Only override intercept if it exists in the formula
    bs_terms <- terms(var_bs_formula)
    if (attr(bs_terms, "intercept") == 1) {
      alpha[1] <- log(0.4 * errv)  # Override intercept
    }
    
    # Restrict alpha values
    for (i in seq_along(alpha)) {
      if (alpha[i] > 1.0) alpha[i] <- 1.0
      if (alpha[i] < -1.0) alpha[i] <- -1.0
    }
  } else {
    alpha <- log(0.4 * errv)
    names(alpha) <- "alpha_intercept"
  }
  
  # Combine all parameters
  params <- c(beta_init, alpha, tau)
  
  # Create parameter names
  beta_names <- names(beta_init)
  alpha_names <- if (is.null(var_bs_formula)) {
    names(alpha)
  } else {
    paste0("alpha_", names(alpha))
  }
  tau_names <- if (is.null(var_ws_formula)) {
    names(tau)
  } else {
    paste0("tau_", names(tau))
  }
  
  param_names <- c(beta_names, alpha_names, tau_names)
  
  list(
    params = params,
    param_names = param_names,
    errv = errv,
    mean_fit = ols_fit
  )
}


#-------------------------------------------------------------------------------


#' Run EM algorithm for initial parameter estimation
#'
#' Refines initial parameter estimates using EM algorithm
#' NOTE: This function assumes a random intercept model
#'
#' @param data Data frame containing model variables
#' @param matrices Design matrices from setup_matrices
#' @param mean_formula Formula for fixed effects mean model
#' @param id_var Name of ID variable column
#' @param response_var Name of response variable column
#' @param init_values List of initial values from init_1
#' @param max_iter Maximum number of EM iterations
#' @return Updated init_values list with refined parameter estimates
run_em <- function(data, matrices, mean_formula, id_var, response_var,
                   init_values, max_iter = 20) {
  # Extract necessary components
  n_fixed <- ncol(matrices$X)
  X <- matrices$X
  y <- data[[response_var]]
  id_factor <- as.factor(data[[id_var]]) # Use a factor for efficient grouping
  n_clusters <- length(levels(id_factor))
  
  # Get cluster sizes efficiently
  cluster_sizes <- as.vector(table(id_factor))
  
  # Extract parameter values
  beta <- init_values$params[1:n_fixed]
  n_alpha <- ncol(matrices$U)
  lbsvar <- init_values$params[n_fixed + 1] # EM only uses the first alpha (intercept)
  tau_start_index <- n_fixed + n_alpha + 1
  lwsvar <- init_values$params[tau_start_index] # EM only uses the first tau
  
  # Pre-allocate
  thetaem <- numeric(n_clusters)  # posterior means
  pvarem <- numeric(n_clusters)   # posterior variances
  
  # EM iterations
  for (iter in 1:max_iter) {
    # E STEP
    rho <- exp(lbsvar) / (exp(lbsvar) + exp(lwsvar))
    
    tempr <- (cluster_sizes * rho) / (1.0 + (cluster_sizes - 1.0) * rho)
    
    pvarem <- exp(lbsvar) * (1.0 - tempr)
    tempr <- tempr / cluster_sizes
    
    pred <- as.vector(X %*% beta)
    
    residuals_vec <- y - pred
    rtemp_vec <- tapply(residuals_vec, id_factor, sum)
    thetaem <- tempr * rtemp_vec
    
    # M STEP
    theta_long <- thetaem[id_factor]
    
    y_adj <- y - theta_long
    
    # Update beta using efficient matrix algebra
    X_transpose <- t(X)
    XtX <- X_transpose %*% X
    XtX_inv <- solve(XtX)
    Xty <- X_transpose %*% y_adj
    beta <- XtX_inv %*% Xty
    
    # Update variance components
    bs_var_new <- (sum(thetaem^2) + sum(pvarem)) / n_clusters
    lbsvar <- log(bs_var_new)
    
    pred_new <- as.vector(X %*% beta)
    pvarem_long <- pvarem[id_factor]
    
    errv <- sum((y - pred_new - theta_long)^2 + pvarem_long)
    lwsvar <- log(errv / length(y))
  }
  
  # Update output parameters
  init_values$params[1:n_fixed] <- beta
  init_values$params[n_fixed + 1] <- lbsvar      # Only update first alpha (intercept)
  init_values$params[tau_start_index] <- lwsvar # Update only first tau
  
  # Store the EB estimates
  init_values$theta_em <- thetaem
  init_values$pv_em <- pvarem
  
  return(init_values)
}


#-------------------------------------------------------------------------------


#' Initialize stage 1 parameters with EM refinement
#'
#' Combines initial OLS-based parameter estimates with EM algorithm refinement
#'
#' @param data Data frame containing model variables
#' @param matrices Design matrices from setup_matrices
#' @param mean_formula Formula for fixed effects mean model
#' @param var_bs_formula Formula for between-subject variance model (or NULL)
#' @param var_ws_formula Formula for within-subject variance model
#' @param id_var Name of ID variable column
#' @param response_var Name of response variable column
#' @return List of initial parameter values refined by EM algorithm
init_1_with_em <- function(data, matrices, mean_formula, 
                           var_bs_formula = NULL, var_ws_formula = NULL,
                           id_var, response_var) {
  # Get initial values using OLS method
  init_vals <- init_1(data, mean_formula, 
                      var_bs_formula, var_ws_formula, response_var)
  
  # Only run EM algorithm for random intercept models
  # EM algorithm is designed for models with an intercept in the BS variance
  if (!is.null(var_bs_formula)) {
    bs_terms <- terms(var_bs_formula)
    has_bs_intercept <- attr(bs_terms, "intercept") == 1
    
    if (has_bs_intercept) {
      # Run EM for random intercept models
      init_vals <- run_em(data, matrices, mean_formula, id_var, response_var, init_vals)
    } else {
      # For no-intercept BS models, skip EM refinement
      cat("Note: Skipping EM refinement for no-intercept between-subject variance model\n")
    }
  }
  return(init_vals)
}


#-------------------------------------------------------------------------------


#' Initialize stage 2 parameters
#'
#' Computes initial values for Stage 2 model parameters based on Stage 1 results
#'
#' @param data Data frame containing model variables
#' @param response_var Name of response variable column
#' @param mean_fit Mean model fit from Stage 1
#' @param tau_intercept tau intercept parameter from Stage 1
#' @param var_ws_formula Formula for within-subject variance model for Stage 2
#' @return Vector of initial tau parameters for Stage 2
#' @importFrom stats lm coef residuals update
#' @export
init_2 <- function(data, response_var, mean_fit, 
                   tau_intercept, var_ws_formula) {
  # Compute residuals from stage 1 model
  r <- residuals(mean_fit)
  log_r2 <- log(r^2 + .Machine$double.eps)
  
  # Add log-squared residuals to data
  ws_data <- data
  ws_data$log_r2 <- log_r2
  
  # Fit WS model again to get slopes
  ws_fit <- lm(update(var_ws_formula, log_r2 ~ .), data = ws_data)
  
  # Extract coefficients but retain tau_intercept from stage 1
  ws_coefs <- coef(ws_fit)
  
  # Additional WS terms (everything except intercept)
  tau_additional <- ws_coefs[-1]
  if (length(tau_additional) == 0) {
    tau_additional <- numeric(0)
  }
  
  # Combine fixed intercept with additional terms
  tau <- c(tau_intercept, tau_additional)
  
  return(tau)
}


#-------------------------------------------------------------------------------


#' Initialize stage 3 parameters
#'
#' Computes initial values for Stage 3 model parameters based on stage 2 results
#'
#' @param data Data frame containing model variables
#' @param id_var Name of ID variable column
#' @param response_var Name of response variable column
#' @param fit2 Results from Stage 2 fit
#' @param model_type Type of model: "independent", "linear", "interaction", or "quadratic"
#' @return Vector of initial parameter values for Stage 3
#' @importFrom dplyr group_by summarize n %>%
#' @importFrom rlang .data
#' @importFrom stats var cor sd
#' @export
init_3 <- function(data, id_var, response_var, fit2, model_type = "linear") {
  # Extract final estimates from stage 2 results using descriptive names
  stage2_params_table <- fit2$results_table
  stage2_estimates <- stage2_params_table$Estimate
  stage2_param_names <- stage2_params_table$Parameter
  
  # Extract parameters based on descriptive names
  beta_indices <- which(!grepl("^(BS_|WS_|Lin_loc|Inter|Quad_loc|Std_Dev)", stage2_param_names))
  beta_final <- stage2_estimates[beta_indices]
  
  alpha_indices <- which(grepl("^BS_", stage2_param_names))
  alpha_final <- stage2_estimates[alpha_indices]
  
  tau_indices <- which(grepl("^WS_", stage2_param_names))
  tau_final <- stage2_estimates[tau_indices]
  
  # Compute subject means and variances
  id_summary <- data %>%
    group_by(.data[[id_var]]) %>%
    summarize(
      m = mean(.data[[response_var]]),
      v = if (n() > 1) var(.data[[response_var]]) else 0,
      .groups = 'drop'
    )
  
  # Compute correlation (rcorr) between subject means and variances
  rcorr <- if (var(id_summary$m) > 0 && var(id_summary$v) > 0) {
    cor(id_summary$m, id_summary$v)
  } else {
    0
  }
  
  # Compute sdlv = standard deviation of log(var+1)
  logvars <- log(id_summary$v + 1)
  sdlv <- sd(logvars)
  
  # Determine parameters based on model_type
  if (model_type == "independent") {
    # One scale parameters: sigma_1=sdlv
    spar <- c(sdlv)
  } else if (model_type == "linear") {
    # Two scale parameters: sigma_1=rcorr, sigma_2=sdlv
    spar <- c(rcorr, sdlv)
  } else if (model_type == "interaction") {
    # Three scale parameters: theta_1, theta_1*theta_2, theta_2
    spar <- c(rcorr, 0, sdlv)  # Initial value of interaction term set to 0
  } else if (model_type == "quadratic") {
    # Three scale parameters: theta_1, theta_1^2, theta_2
    spar <- c(rcorr, -rcorr/4, sdlv)  # Initial value of quadratic term
  } else {
    stop("Invalid model_type specified. Use 'independent', 'linear', 'interaction', or 'quadratic'.")
  }
  
  # Combine Stage 2 final parameters with spar for stage 3 start values
  stage3_start_params <- c(beta_final, alpha_final, tau_final, spar)
  
  return(stage3_start_params)
}
