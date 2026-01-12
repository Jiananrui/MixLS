# R/mels_loglike.R


#-------------------------------------------------------------------------------


#' Stage 1 & 2 log-likelihood function
#'
#' Wrapper for the C++ function that calculates the log-likelihood for the 
#' mixed effects location-scale model, supporting adaptive quadrature.
#'
#' @param params Vector of parameters (beta, alpha, tau)
#' @param model_data Prepared model data from prepare_model_data function
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @param eb_theta Empirical Bayes estimates for subject means (for adaptive)
#' @param eb_sd Empirical Bayes estimates for subject standard deviations (for adaptive)
#' @return Log-likelihood value
#' @export
stage12_loglik_wrapper <- function(params, model_data, adaptive = 1, eb_theta = NULL, eb_sd = NULL) {
  if (adaptive == 1 && !is.null(eb_theta) && !is.null(eb_sd)) {
    # Call the C++ function for adaptive log-likelihood
    loglik <- stage12_loglik_adaptive(
      params = params,
      X = model_data$matrices$X,
      U = model_data$matrices$U,
      Z = model_data$matrices$Z,
      y = model_data$y,
      id = model_data$id_numeric,
      points = model_data$gh$points,
      weights = model_data$gh$weights,
      adaptive = 1,
      eb_theta = eb_theta,
      eb_sd = eb_sd
    )
  } else {
    # Call the C++ function for fixed quadrature log-likelihood
    loglik <- stage12_loglik_rcpp(
      params = params,
      X = model_data$matrices$X,
      U = model_data$matrices$U,
      Z = model_data$matrices$Z,
      y = model_data$y,
      id = model_data$id_numeric,
      points = model_data$gh$points,
      weights = model_data$gh$weights
    )
  }
  return(loglik)
}


#-------------------------------------------------------------------------------


#' Stage 1 & 2 gradient function
#'
#' Wrapper for C++ gradient function, supporting adaptive quadrature
#'
#' @param params Vector of parameters (beta, alpha, tau)
#' @param model_data Prepared model data from prepare_model_data function
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @param eb_theta Empirical Bayes estimates for each subject (only used if adaptive = 1)
#' @param eb_sd Posterior standard deviations for each subject (only used if adaptive = 1)
#' @return Gradient vector
#' @export
stage12_gradient_wrapper <- function(params, model_data, adaptive = 1, eb_theta = NULL, eb_sd = NULL) {
  if (adaptive == 1) {
    # Call the C++ function for adaptive gradient
    gradient <- stage12_gradient_adaptive(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      adaptive,
      eb_theta,
      eb_sd
    )
  } else {
    # Call the C++ function for fixed gradient
    gradient <- stage12_gradient_rcpp(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights
    )
  }
  return(gradient)
}


#-------------------------------------------------------------------------------


#' Stage 1 & 2 hessian function
#'
#' Wrapper for C++ Hessian function, supporting adaptive quadrature
#'
#' @param params Vector of parameters (beta, alpha, tau)
#' @param model_data Prepared model data from prepare_model_data function
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @param eb_theta Empirical Bayes estimates for each subject (only used if adaptive = 1)
#' @param eb_sd Posterior standard deviations for each subject (only used if adaptive = 1)
#' @return Hessian matrix
#' @export
stage12_hessian_wrapper <- function(params, model_data, adaptive = 1, eb_theta = NULL, eb_sd = NULL) {
  if (adaptive == 1) {
    # Call the C++ function for adaptive Hessian
    hessian <- stage12_hessian_adaptive(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      adaptive,
      eb_theta,
      eb_sd  
    )
  } else {
    # Call the C++ function for fixed Hessian
    hessian <- stage12_hessian_rcpp(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights
    )
  }
  return(hessian)
}


#-------------------------------------------------------------------------------


#' Stage 3 log-likelihood function
#'
#' Wrapper for C++ log-likelihood function, supporting adaptive quadrature for the
#' bivariate random effects model.
#'
#' @param params Vector of parameters (beta, alpha, tau, sigma)
#' @param model_data Prepared model data from prepare_model_data function
#' @param model_type Type of model: "independent", "linear", "interaction", or "quadratic"
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @param eb_theta2 Matrix of posterior means (n_subjects x 2) for the two random effects.
#' @param eb_thetav Matrix of posterior VCV components (n_subjects x 3).
#' @return Log-likelihood value
#' @export
stage3_loglik_wrapper <- function(params, model_data, model_type,
                                  adaptive = 1, eb_theta2 = NULL, eb_thetav = NULL) {
  
  if (adaptive == 1 && !is.null(eb_theta2) && !is.null(eb_thetav)) {
    # Call the C++ function for adaptive log-likelihood with bivariate EB estimates
    loglik <- stage3_loglik_adaptive_combined(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      model_type,
      adaptive,
      eb_theta2,
      eb_thetav
    )
  } else {
    # Call the original C++ function for fixed quadrature
    loglik <- stage3_loglik_combined(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      model_type
    )
  }
  return(loglik)
}


#-------------------------------------------------------------------------------


#' Stage 3 Gradient Function
#'
#' Wrapper for C++ gradient function, supporting adaptive quadrature for the
#' bivariate random effects model.
#'
#' @param params Vector of parameters (beta, alpha, tau, sigma)
#' @param model_data Prepared model data from prepare_model_data function
#' @param model_type Type of model: "independent", "linear", "interaction", or "quadratic"
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @param eb_theta2 Matrix of posterior means (n_subjects x 2) for the two random effects.
#' @param eb_thetav Matrix of posterior VCV components (n_subjects x 3).
#' @return Gradient vector
#' @export
stage3_gradient_wrapper <- function(params, model_data, model_type,
                                    adaptive = 1, eb_theta2 = NULL, eb_thetav = NULL) {
  
  if (adaptive == 1 && !is.null(eb_theta2) && !is.null(eb_thetav)) {
    # Call the C++ function for adaptive gradient with bivariate EB estimates
    gradient <- stage3_gradient_adaptive_combined(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      model_type,
      adaptive,
      eb_theta2,
      eb_thetav
    )
  } else {
    # Call the original C++ function for fixed gradient
    gradient <- stage3_gradient_combined(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      model_type
    )
  }
  return(gradient)
}


#-------------------------------------------------------------------------------


#' Stage 3 Hessian Function (Wrapper)
#'
#' Wrapper for C++ Hessian function, supporting adaptive quadrature for the
#' bivariate random effects model.
#'
#' @param params Vector of parameters (beta, alpha, tau, sigma)
#' @param model_data Prepared model data from prepare_model_data function
#' @param model_type Type of model: "independent", "linear", "interaction", or "quadratic"
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive)
#' @param eb_theta2 Matrix of posterior means (n_subjects x 2) for the two random effects.
#' @param eb_thetav Matrix of posterior VCV components (n_subjects x 3).
#' @return Hessian matrix
#' @export
stage3_hessian_wrapper <- function(params, model_data, model_type,
                                   adaptive = 1, eb_theta2 = NULL, eb_thetav = NULL) {
  
  if (adaptive == 1 && !is.null(eb_theta2) && !is.null(eb_thetav)) {
    # Call the C++ function for adaptive Hessian with bivariate EB estimates
    hessian <- stage3_hessian_adaptive_combined(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      model_type,
      adaptive,
      eb_theta2,
      eb_thetav 
    )
  } else {
    # Call the original C++ function for fixed Hessian
    hessian <- stage3_hessian_combined(
      params,
      model_data$matrices$X,
      model_data$matrices$U,
      model_data$matrices$Z,
      model_data$y,
      model_data$id_numeric,
      model_data$gh$points,
      model_data$gh$weights,
      model_type
    )
  }
  return(hessian)
}

