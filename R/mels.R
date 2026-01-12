# R/mels.R


#-------------------------------------------------------------------------------


#' Fit Mixed-Effects Location Scale Model
#'
#' Main function to fit a mixed-effects location scale model with heterogeneous within-subject variances
#'
#' @param data Data frame containing all variables
#' @param fixed Formula for the fixed effects (mean) model
#' @param bs Formula for the between-subject variance model
#' @param ws Formula for the within-subject variance model (used for Stage 2 & 3)
#' @param id Name of the column containing subject identifiers
#' @param response Name of the column containing response values
#' @param nq Number of quadrature points (default: 11)
#' @param maxiter Maximum number of iterations for optimization (default: 200)
#' @param tol Convergence tolerance (default: 1e-5)
#' @param stage Which stages to fit: 1, 2, or 3. Stage 2 fits 1 & 2. Stage 3 fits 1, 2, & 3. (default: 3)
#' @param type A character vector specifying Stage 3 models to fit (e.g., "linear", c("linear", "quadratic")). Use "all" to fit all types.
#' @param ridge1 Stage 1 initial ridge parameter value (default: 0)
#' @param ridge2 Stage 2 initial ridge parameter value (default: 0.1)
#' @param ridge3 Stage 3 initial ridge parameter value (default: 0.2)
#' @param adaptive Use adaptive quadrature (0 = fixed, 1 = adaptive) (default: 0)
#' @param verbose Logical. If TRUE, print iteration details during fitting. If FALSE, suppress output (default: FALSE)
#' @return A mels object containing model results
#' @useDynLib MixLS, .registration = TRUE
#' @importFrom Rcpp sourceCpp
#' @importFrom RcppParallel RcppParallelLibs
#' @importFrom stats as.formula terms pchisq pnorm
#' @export
mels <- function(data, id, response,
                     fixed, bs, ws,
                     nq = 11, maxiter = 200, tol = 1e-5,
                     stage = 3, type = "linear",
                     ridge1 = 0,
                     ridge2 = 0.1,
                     ridge3 = 0.2,
                     adaptive = 1,
                     verbose = TRUE) {

  if (!verbose) {
    sink(nullfile())
    on.exit(sink(), add = TRUE)
  }

  # Input validation
  if (!inherits(fixed, "formula") && !is.null(fixed)) {
    stop("fixed must be a formula or NULL")
  }
  if (!inherits(bs, "formula")) {
    stop("bs must be a formula")
  }
  if (!inherits(ws, "formula")) {
    stop("ws must be a formula")
  }
  if (!(id %in% names(data))) {
    stop(paste("id", id, "not found in data"))
  }
  if (!(response %in% names(data))) {
    stop(paste("response", response, "not found in data"))
  }
  if (!(stage %in% c(1, 2, 3))) {
    stop("stage must be one of 1, 2, or 3.")
  }
  if (!(adaptive %in% c(0, 1))) {
    stop("adaptive must be 0 (fixed) or 1 (adaptive)")
  }

  valid_types <- c("independent", "linear", "interaction", "quadratic")
  if (identical(type, "all")) {
    type <- valid_types
  } else if (!all(type %in% valid_types)) {
    stop("type must be 'all' or a vector containing 'independent', 'linear', 'interaction', 'quadratic'")
  }

  # Check if Stage 2 should be skipped
  # Stage 2 is skipped if ws is ~ 1 (same as Stage 1's fixed formula)
  skip_stage2 <- paste(deparse(ws), collapse = "") == deparse("~ 1")

  if (skip_stage2 && stage == 2) {
    warning("Stage 2 requested but ws is ~ 1 (same as Stage 1). Setting stage = 1.")
    stage <- 1
  }

  # Print stage plan for user information
  if (skip_stage2) {
    cat("Stage plan: Stage 1 -> Stage 3, Stage 2 skipped \n")
  } else {
    cat("Stage plan: Stage 1 -> Stage 2 -> Stage 3 \n")
  }

  data <- data[order(data[[id]]), ]

  # Initialize result object
  result <- list(
    call = match.call(),
    data_summary = list(
      n_obs = nrow(data),
      n_subjects = length(unique(data[[id]])),
      fixed = fixed,
      bs = bs,
      ws_stage1 = ~ 1,  # Always ~ 1 for Stage 1
      ws = ws,  # User-provided formula for Stages 2 & 3
      skip_stage2 = skip_stage2
    )
  )

  # Fit Stage 1 (always uses ~ 1 for within-subject variance)
  cat("\n========== STAGE 1 ==========\n")
  cat("Note: Stage 1 always uses ~ 1 for within-subject variance (for starting values)\n")
  stage1_results <- stage1_fit(
    data = data,
    fixed = fixed,
    bs = bs,
    ws = ~ 1,  # Fixed: always use ~ 1 for Stage 1
    id = id,
    response = response,
    nq = nq,
    maxiter = maxiter,
    tol = tol,
    init_ridge = ridge1,
    adaptive = adaptive
  )
  result$stage1 <- stage1_results

  # Fit Stage 2 (only if not skipped and stage >= 2)
  if (stage >= 2 && !skip_stage2) {
    cat("\n========== STAGE 2 ==========\n")
    stage2_results <- stage2_fit(
      stage1_result = result$stage1,
      data = data,
      fixed = fixed,
      bs = bs,
      ws = ws,  # Use user-provided formula
      id = id,
      response = response,
      nq = nq,
      maxiter = maxiter,
      tol = tol,
      init_ridge = ridge2,
      adaptive = adaptive
    )

    # Perform LRT between Stage 1 and Stage 2
    if (!is.null(result$stage1) && !is.null(stage2_results)) {
      lrt_stat <- 2 * (stage2_results$final_loglik - result$stage1$final_loglik)
      df_diff <- length(stage2_results$results_table$Estimate) - length(result$stage1$results_table$Estimate)
      if (df_diff > 0) {
        p_value <- pchisq(lrt_stat, df = df_diff, lower.tail = FALSE)
        stage2_results$lrt_vs_stage1 <- data.frame(
          LRT_statistic = lrt_stat,
          df = df_diff,
          p_value = p_value
        )
      }
    }
    result$stage2 <- stage2_results
  } else if (stage >= 2 && skip_stage2) {
    cat("\n========== STAGE 2 SKIPPED ==========\n")
    cat("Stage 2 skipped because ws is ~ 1 (same as Stage 1)\n")
  }

  # Fit Stage 3
  if (stage >= 3) {
    # Determine which result to use as base for Stage 3
    stage_for_stage3 <- if (!skip_stage2 && !is.null(result$stage2)) {
      result$stage2
    } else {
      result$stage1
    }

    if (is.null(stage_for_stage3)) {
      stop("Cannot fit Stage 3 without a valid base stage result.")
    }

    stage3_results_list <- list()
    # Loop to fit all requested Stage 3 models
    for (model in type) {
      cat(paste0("\n========== STAGE 3 (", model, ") ==========\n"))
      stage3_results_list[[model]] <- stage3_fit(
        stage2_result = stage_for_stage3,  # Use appropriate base stage
        data = data, fixed = fixed, bs = bs,
        ws = ws, id = id, response = response,
        nq = nq, maxiter = maxiter, tol = tol, model_type = model,
        init_ridge = ridge3, adaptive = adaptive
      )
    }

    # Process Stage 3 results
    if (length(stage3_results_list) == 1) {
      result$stage3 <- stage3_results_list[[1]]
    } else {
      result$types <- stage3_results_list

      # Create AIC/BIC comparison table
      comparison_df <- data.frame(
        Model = names(stage3_results_list),
        LogLikelihood = sapply(stage3_results_list, function(x) x$final_loglik),
        Parameters = sapply(stage3_results_list, function(x) length(x$results_table$Estimate)),
        AIC = sapply(stage3_results_list, function(x) x$AIC),
        BIC = sapply(stage3_results_list, function(x) x$BIC)
      )

      # Find best model by AIC and mark it
      best_model_name <- comparison_df$Model[which.min(comparison_df$AIC)]
      result$stage3_best <- stage3_results_list[[best_model_name]]
      comparison_df$BestModel <- comparison_df$Model == best_model_name
      result$stage3_comparison <- comparison_df

      # Perform LRT for nested Stage 3 models
      lrt_results <- list()
      # Helper for LRT
      perform_lrt <- function(mod1, mod2, name1, name2) {
        if (name1 %in% names(stage3_results_list) && name2 %in% names(stage3_results_list)) {
          lrt_stat <- 2 * (mod2$final_loglik - mod1$final_loglik)
          df_diff <- length(mod2$results_table$Estimate) - length(mod1$results_table$Estimate)
          if (df_diff > 0) {
            return(data.frame(Model1=name1, Model2=name2, LRT_statistic=lrt_stat, df=df_diff,
                              p_value=pchisq(lrt_stat, df=df_diff, lower.tail=FALSE)))
          }
        }
        return(NULL)
      }

      lrt_results[["Lin_vs_Ind"]] <- perform_lrt(stage3_results_list[["independent"]], stage3_results_list[["linear"]], "independent", "linear")
      lrt_results[["Int_vs_Lin"]] <- perform_lrt(stage3_results_list[["linear"]], stage3_results_list[["interaction"]], "linear", "interaction")
      lrt_results[["Qua_vs_Lin"]] <- perform_lrt(stage3_results_list[["linear"]], stage3_results_list[["quadratic"]], "linear", "quadratic")

      lrt_df <- do.call(rbind, lrt_results)
      if (!is.null(lrt_df)) {
        row.names(lrt_df) <- NULL
        result$stage3_lrt_comparison <- lrt_df
      }
    }
  }

  # Perform LRT between best Stage 3 model and the base stage (Stage 2 or Stage 1)
  final_type <- if (!is.null(result$stage3_best)) result$stage3_best else result$stage3
  base_stage_for_lrt <- if (!skip_stage2 && !is.null(result$stage2)) result$stage2 else result$stage1

  if (!is.null(base_stage_for_lrt) && !is.null(final_type)) {
    lrt_stat_s3 <- 2 * (final_type$final_loglik - base_stage_for_lrt$final_loglik)
    df_diff_s3 <- length(final_type$results_table$Estimate) - length(base_stage_for_lrt$results_table$Estimate)
    if (df_diff_s3 > 0) {
      p_value_s3 <- pchisq(lrt_stat_s3, df = df_diff_s3, lower.tail = FALSE)
      lrt_data <- data.frame(LRT_statistic = lrt_stat_s3, df = df_diff_s3, p_value = p_value_s3)

      if (!is.null(result$stage3_best)) {
        result$stage3_best$lrt_vs_base <- lrt_data
      } else if (!is.null(result$stage3)) {
        result$stage3$lrt_vs_base <- lrt_data
      }
    }
  }

  # Set class for custom methods
  class(result) <- "mels"
  return(result)
}

