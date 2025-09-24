# R/methods.R


#-------------------------------------------------------------------------------


#' Print method for mixregls objects
#'
#' @param x A mixregls object
#' @param ... Additional arguments (not used)
#' @export
print.mixregls <- function(x, ...) {
  cat("Mixed-Effects Location Scale Model\n\n")
  cat("Call:\n")
  print(x$call)
  cat("\n")

  cat("Data Summary:\n")
  cat("  Number of observations:", x$data_summary$n_obs, "\n")
  cat("  Number of subjects:", x$data_summary$n_subjects, "\n")
  cat("\n")

  cat("Model Formulas:\n")
  cat("  Mean model:", deparse(x$data_summary$mean_formula), "\n")
  cat("  Between-subject variance:", deparse(x$data_summary$var_bs_formula), "\n")
  cat("  Within-subject variance (Stage 1):", deparse(x$data_summary$var_ws_formula_stage1), "\n")
  cat("  Within-subject variance (Stages 2 & 3):", deparse(x$data_summary$var_ws_formula), "\n")

  if (x$data_summary$skip_stage2) {
    cat("  Note: Stage 2 skipped (same formula as Stage 1)\n")
  }
  cat("\n")

  # Print results for the highest stage fitted
  if (!is.null(x$stage3) || !is.null(x$stage3_best)) {
    if (!is.null(x$stage3_comparison)) {
      cat("Stage 3 Model Comparison:\n")
      print(x$stage3_comparison)
      cat("\n")
      cat("Best Stage 3 Model Results:\n")
      print(x$stage3_best$results_table)
      cat("\n")
      cat("Log-likelihood:", x$stage3_best$final_loglik, "\n")
      cat("Convergence:", x$stage3_best$convergence, "\n")
    } else {
      cat("Stage 3 Results:\n")
      print(x$stage3$results_table)
      cat("\n")
      cat("Log-likelihood:", x$stage3$final_loglik, "\n")
      cat("Convergence:", x$stage3$convergence, "\n")
    }
  } else if (!is.null(x$stage2)) {
    cat("Stage 2 Results:\n")
    print(x$stage2$results_table)
    cat("\n")
    cat("Log-likelihood:", x$stage2$final_loglik, "\n")
    cat("Convergence:", x$stage2$convergence, "\n")
  } else if (!is.null(x$stage1)) {
    cat("Stage 1 Results:\n")
    print(x$stage1$results_table)
    cat("\n")
    cat("Log-likelihood:", x$stage1$final_loglik, "\n")
    cat("Convergence:", x$stage1$convergence, "\n")
    if (!is.null(x$stage1$adaptive_used) && x$stage1$adaptive_used == 1) {
      cat("Quadrature: Adaptive (empirical Bayes centered)\n")
      if (!is.null(x$stage1$standardized_residuals)) {
        cat(sprintf("\nStandardized Residuals Summary:\n"))
        cat(sprintf("  Range: [%.4f, %.4f]\n", min(x$stage1$standardized_residuals), max(x$stage1$standardized_residuals)))
        cat(sprintf("  Mean: %.6f, SD: %.6f\n", mean(x$stage1$standardized_residuals), sd(x$stage1$standardized_residuals)))
      }
    } else {
      cat("Quadrature: Fixed (standard Gauss-Hermite)\n")
    }
  }
}


#-------------------------------------------------------------------------------


#' Summary method for mixregls objects
#'
#' @param object A mixregls object
#' @param ... Additional arguments (not used)
#' @export
summary.mixregls <- function(object, ...) {
  result <- object
  class(result) <- "summary.mixregls"
  return(result)
}


#-------------------------------------------------------------------------------


#' Print method for summary.mixregls objects
#'
#' @param x A summary.mixregls object
#' @param ... Additional arguments (not used)
#' @export
print.summary.mixregls <- function(x, ...) {
  cat("==========================================\n")
  cat(" Mixed-Effects Location Scale Model Summary\n")
  cat("==========================================\n\n")

  cat("Model Fit Information:\n")

  if (!is.null(x$stage1)) {
    cat("  Stage 1:\n")
    cat("    Log-likelihood:", x$stage1$final_loglik, "\n")
    cat("    AIC:", x$stage1$AIC, "\n")
    cat("    BIC:", x$stage1$BIC, "\n")
    cat("    Iterations:", x$stage1$iterations, "\n")
  }

  if (!is.null(x$stage2)) {
    cat("  Stage 2:\n")
    cat("    Log-likelihood:", x$stage2$final_loglik, "\n")
    cat("    AIC:", x$stage2$AIC, "\n")
    cat("    BIC:", x$stage2$BIC, "\n")
    cat("    Iterations:", x$stage2$iterations, "\n")
  } else if (x$data_summary$skip_stage2) {
    cat("  Stage 2: Skipped (same formula as Stage 1)\n")
  }

  best_s3_model <- if (!is.null(x$stage3_best)) x$stage3_best else x$stage3

  if (!is.null(x$stage3_comparison)) {
    cat("\nStage 3 Model Comparison:\n")
    print(x$stage3_comparison)

    if (!is.null(x$stage3_lrt_comparison)) {
      cat("\nStage 3 Nested Model LRT Comparison:\n")
      print(x$stage3_lrt_comparison)
    }

    cat("\n  Best Stage 3 model:", x$stage3_best$model_type, "\n")
    cat("    Log-likelihood:", x$stage3_best$final_loglik, "\n")
    cat("    AIC:", x$stage3_best$AIC, "\n")
    cat("    BIC:", x$stage3_best$BIC, "\n")
    cat("    Iterations:", x$stage3_best$iterations, "\n")
  } else if (!is.null(x$stage3)) {
    cat("  Stage 3:\n")
    cat("    Model type:", x$stage3$model_type, "\n")
    cat("    Log-likelihood:", x$stage3$final_loglik, "\n")
    cat("    AIC:", x$stage3$AIC, "\n")
    cat("    BIC:", x$stage3$BIC, "\n")
    cat("    Iterations:", x$stage3$iterations, "\n")
  }
  cat("\n")

  # --- Inter-Stage Likelihood Ratio Tests ---
  cat("Likelihood Ratio Tests:\n")
  if(!is.null(x$stage2) && !is.null(x$stage2$lrt_vs_stage1)){
    cat("  Stage 2 vs. Stage 1:\n")
    cat("    Chi-sq:", x$stage2$lrt_vs_stage1$LRT_statistic,
        "on", x$stage2$lrt_vs_stage1$df, "DF, ",
        "p-value:", x$stage2$lrt_vs_stage1$p_value, "\n")
  }

  # Handle LRT for Stage 3 vs base stage (could be Stage 1 or Stage 2)
  base_stage_name <- if (!is.null(x$stage2)) "Stage 2" else "Stage 1"
  lrt_vs_base <- if (!is.null(best_s3_model$lrt_vs_base)) {
    best_s3_model$lrt_vs_base
  } else if (!is.null(best_s3_model$lrt_vs_stage2)) {
    best_s3_model$lrt_vs_stage2
  } else {
    NULL
  }

  if(!is.null(lrt_vs_base)){
    cat("  Stage 3 vs.", base_stage_name, ":\n")
    cat("    Chi-sq:", lrt_vs_base$LRT_statistic,
        "on", lrt_vs_base$df, "DF, ",
        "p-value:", lrt_vs_base$p_value, "\n")
  }
  cat("\n")

  cat("Model Formulas:\n")
  cat("  Mean model:", deparse(x$data_summary$mean_formula), "\n")
  cat("  Between-subject variance:", deparse(x$data_summary$var_bs_formula), "\n")
  cat("  Within-subject variance (Stage 1):", deparse(x$data_summary$var_ws_formula_stage1), "\n")
  cat("  Within-subject variance (Stages 2 & 3):", deparse(x$data_summary$var_ws_formula), "\n")
  if (x$data_summary$skip_stage2) {
    cat("  Note: Stage 2 skipped (same formula as Stage 1)\n")
  }
  cat("\n")

  # Print detailed parameter tables for each stage
  if (!is.null(x$stage1)) {
    cat("Stage 1 Parameters:\n")
    print(x$stage1$results_table)
    cat("\n")
  }

  if (!is.null(x$stage2)) {
    cat("Stage 2 Parameters:\n")
    print(x$stage2$results_table)
    cat("\n")
  }

  if (!is.null(x$stage3_best)) {
    cat("Best Stage 3 Parameters (", x$stage3_best$model_type, "):\n", sep="")
    print(x$stage3_best$results_table)
  } else if (!is.null(x$stage3)) {
    cat("Stage 3 Parameters (", x$stage3$model_type, "):\n", sep="")
    print(x$stage3$results_table)
  }
}
