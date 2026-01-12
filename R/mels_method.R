# R/mels_methods.R


#-------------------------------------------------------------------------------


#' Print method for mels objects
#'
#' @param x A mels object
#' @param ... Additional arguments
#' @export
print.mels <- function(x, ...) {
  cat("Mixed-Effects Location Scale Model\n")
  cat("Call:\n")
  print(x$call)
  cat("\n")
  
  cat("Data Summary:\n")
  cat("  Number of observations:", x$data_summary$n_obs, "\n")
  cat("  Number of subjects:    ", x$data_summary$n_subjects, "\n")
  cat("\n")
  
  cat("Model Formulas:\n")
  cat("  Mean model:              ", deparse(x$data_summary$fixed), "\n")
  cat("  Between-subject variance:", deparse(x$data_summary$bs), "\n")
  cat("  Within-subject variance: ", deparse(x$data_summary$ws), "\n")
  
  # Check for stage 3 comparisons (AIC/BIC)
  if (!is.null(x$stage3_comparison)) {
    cat("\n")
    cat("--------------------------------------------------\n")
    cat("Stage 3 Model Comparison (Sorted by AIC)\n")
    cat("--------------------------------------------------\n")
    
    # Format the comparison table
    comp_df <- x$stage3_comparison
    # Sort by AIC
    comp_df <- comp_df[order(comp_df$AIC), ]
    
    # Define widths for columns
    cat(sprintf("%-12s %10s %5s %10s %10s\n", "Model", "LogLik", "Par", "AIC", "BIC"))
    cat(sprintf("%-12s %10s %5s %10s %10s\n", "-----------", "----------", "---", "----------", "----------"))
    
    for(i in 1:nrow(comp_df)) {
      cat(sprintf("%-12s %10.3f %5d %10.3f %10.3f%s\n", 
                  comp_df$Model[i], 
                  comp_df$LogLikelihood[i], 
                  comp_df$Parameters[i], 
                  comp_df$AIC[i], 
                  comp_df$BIC[i],
                  if(comp_df$BestModel[i]) " *" else ""))
    }
    cat("(* indicates best model by AIC)\n")
  }
  
  # Check for stage 3 nested LRTs
  if (!is.null(x$stage3_lrt_comparison)) {
    cat("\n")
    cat("--------------------------------------------------\n")
    cat("Likelihood Ratio Tests (Nested Models)\n")
    cat("--------------------------------------------------\n")
    
    lrt_df <- x$stage3_lrt_comparison
    
    cat(sprintf("%-12s vs %-12s %10s %3s %10s\n", "Base", "Complex", "ChiSq", "DF", "p-value"))
    cat(sprintf("%-12s    %-12s %10s %3s %10s\n", "----", "-------", "-----", "--", "-------"))
    
    for(i in 1:nrow(lrt_df)) {
      cat(sprintf("%-12s vs %-12s %10.4f %3d %10.5f\n",
                  lrt_df$Model1[i],
                  lrt_df$Model2[i],
                  lrt_df$LRT_statistic[i],
                  lrt_df$df[i],
                  lrt_df$p_value[i]))
    }
  }
  
  cat("\n(Use summary() to view detailed parameters for the best fit)\n")
}


#-------------------------------------------------------------------------------


#' Summary method for mels objects
#'
#' @param object A mels object
#' @param ... Additional arguments
#' @export
summary.mels <- function(object, ...) {
  result <- object
  class(result) <- "summary.mels"
  return(result)
}


#-------------------------------------------------------------------------------


#' Print method for summary.mels objects
#'
#' @param x A summary.mels object
#' @param ... Additional arguments (not used)
#' @export
print.summary.mels <- function(x, ...) {
  
  print_stage_sas_style <- function(stage_obj, title, n_obs, n_subjects, model_type = NULL) {
    if (is.null(stage_obj)) return()
    
    # Extract metrics
    ll <- stage_obj$final_loglik
    k <- nrow(stage_obj$results_table)
    iter <- stage_obj$iterations
    ridge <- if(!is.null(stage_obj$final_ridge)) stage_obj$final_ridge else 0.0
    
    # Adaptive status
    is_adaptive <- if (!is.null(stage_obj$adaptive_used) && stage_obj$adaptive_used == 1) "TRUE" else "FALSE"
    
    # Convergence status
    conv_val <- stage_obj$convergence
    is_converged <- "FALSE"
    if (isTRUE(conv_val) || identical(conv_val, "Converged") || (is.numeric(conv_val) && conv_val == 0)) {
      is_converged <- "TRUE"
    } else if (is.character(conv_val)) {
      is_converged <- if (grepl("Converged", conv_val, ignore.case = TRUE)) "TRUE" else "FALSE"
    }
    
    # Top block calculations
    aic_top <- ll - k
    bic_top <- ll - 0.5 * k * log(n_subjects)
    
    # Bottom block calculations
    neg2ll <- -2 * ll
    aic_bot <- neg2ll + 2 * k
    bic_bot <- neg2ll + k * log(n_subjects)
    
    # Print header block
    sep_line <- paste0(rep("-", nchar(title)), collapse = "")
    cat(sep_line, "\n")
    cat(title, "\n")
    cat(sep_line, "\n")
    
    cat(sprintf(" Total  Iterations = %d\n", iter))
    cat(sprintf(" Final Ridge Value = %.1f\n", ridge))
    cat(sprintf(" Adaptive Quad     = %s\n", is_adaptive))
    cat(sprintf(" Converged         = %s\n", is_converged))
    
    if (!is.null(model_type)) {
      cat(sprintf(" Model Type        = %s\n", model_type))
    }
    cat("\n")
    
    # Stats Table
    cat(sprintf(" Log Likelihood                 = %10.3f\n", ll))
    cat(sprintf(" Akaike's Information Criterion = %10.3f\n", aic_top))
    cat(sprintf(" Schwarz's Bayesian Criterion   = %10.3f\n", bic_top))
    cat("\n")
    cat(" ==> multiplied by -2             \n")
    cat(sprintf(" Log Likelihood                 = %10.3f\n", neg2ll))
    cat(sprintf(" Akaike's Information Criterion = %10.3f\n", aic_bot))
    cat(sprintf(" Schwarz's Bayesian Criterion   = %10.3f\n", bic_bot))
    cat("\n\n")
    
    # Parameter tables
    res <- stage_obj$results_table
    
    # Helper to clean names for standard vars
    clean_name_standard <- function(n) {
      n <- gsub("intercept", "Intercpt", n, ignore.case = TRUE)
      n <- gsub("^BS_", "", n)
      n <- gsub("^WS_", "", n)
      return(n)
    }
    
    # Helper to print the table header
    print_table_header <- function() {
      cat(sprintf(" %-15s %12s %12s %12s %12s\n", "Variable", "Estimate", "AsymStdError", "z-value", "p-value"))
      cat(sprintf(" %-15s %12s %12s %12s %12s\n", "--------", "------------", "------------", "------------", "------------"))
    }
    
    # Helper to print rows with custom name cleaning
    print_rows <- function(rows, name_map_func = clean_name_standard) {
      if (nrow(rows) == 0) return()
      for (i in 1:nrow(rows)) {
        nm <- name_map_func(rows$Parameter[i])
        cat(sprintf(" %-15s %12.5f %12.5f %12.5f %12.5f\n",
                    nm,
                    rows$Estimate[i],
                    rows$SE[i],
                    rows$`Z-score`[i],
                    rows$`P-value`[i]))
      }
    }
    
    # Identify special Stage 3 params
    is_lin  <- res$Parameter == "Lin_loc"
    is_quad <- res$Parameter == "Quad_loc"
    is_int  <- res$Parameter == "Inter"
    is_special_s3 <- is_lin | is_quad | is_int
    
    is_bs <- grepl("^BS_", res$Parameter)
    is_ws <- grepl("^WS_", res$Parameter)
    is_std <- grepl("Std_Dev", res$Parameter) | grepl("Random scale", res$Parameter, ignore.case=TRUE)
    
    beta_rows  <- res[!is_bs & !is_ws & !is_std & !is_special_s3, ]
    alpha_rows <- res[is_bs, ]
    tau_rows   <- res[is_ws & !is_special_s3, ] # Exclude special params from standard TAU block
    std_rows   <- res[is_std, ]
    
    # Print BETA
    if (nrow(beta_rows) > 0) {
      cat(" BETA (regression coefficients)\n")
      print_table_header()
      print_rows(beta_rows)
      cat("\n")
    }
    
    # Print ALPHA
    if (nrow(alpha_rows) > 0) {
      cat(" ALPHA (BS variance parameters: log-linear model)\n")
      print_table_header()
      print_rows(alpha_rows)
      cat("\n")
    }
  
    # Print TAU
    if (nrow(tau_rows) > 0) {
      cat(" TAU (WS variance parameters: log-linear model)\n")
      print_table_header()
      print_rows(tau_rows)
      cat("\n")
    }
    
    # 4. Print (Linear / Quadratic / Interaction)
    if (!is.null(model_type)) {
      
      # Case: QUADRATIC
      if (model_type == "quadratic") {
        quad_subset <- res[is_lin | is_quad, ]
        if (nrow(quad_subset) > 0) {
          cat(" Random linear and quadratic location (mean) effects on WS variance\n")
          print_table_header()
          # Custom namer for quadratic block
          quad_namer <- function(n) {
            if (n == "Lin_loc") return("Lin Loc")
            if (n == "Quad_loc") return("Quad Loc")
            return(n)
          }
          print_rows(quad_subset, quad_namer)
          cat("\n")
        }
      }
      
      # Case: LINEAR
      else if (model_type == "linear") {
        lin_subset <- res[is_lin, ]
        if (nrow(lin_subset) > 0) {
          cat(" Random location (mean) effect on WS variance\n")
          print_table_header()
          # Custom namer for linear block
          lin_namer <- function(n) {
            if (n == "Lin_loc") return("Loc Eff")
            return(n)
          }
          print_rows(lin_subset, lin_namer)
          cat("\n")
        }
      }
      
      # Case: INTERACTION
      else if (model_type == "interaction") {
        # Block 1: Linear Loc
        lin_subset <- res[is_lin, ]
        if (nrow(lin_subset) > 0) {
          cat(" Random linear location (mean) effect on WS variance\n")
          print_table_header()
          lin_int_namer <- function(n) { if(n=="Lin_loc") "Lin loc" else n }
          print_rows(lin_subset, lin_int_namer)
          cat("\n")
        }
        
        # Interaction
        int_subset <- res[is_int, ]
        if (nrow(int_subset) > 0) {
          cat(" Random (location & scale) interaction effect on WS variance\n")
          print_table_header()
          int_namer <- function(n) { if(n=="Inter") "Inter" else n }
          print_rows(int_subset, int_namer)
          cat("\n")
        }
      }
      
      # Case: INDEPENDENT
      else if (any(is_special_s3) && model_type == "independent") {
        # Fallback if specific params exist but type is independent
        spec_subset <- res[is_special_s3, ]
        cat(" Additional Model Parameters\n")
        print_table_header()
        print_rows(spec_subset)
        cat("\n")
      }
    }
    
    # 5. Print random scale
    if (nrow(std_rows) > 0) {
      cat(" Random scale standard deviation\n")
      print_table_header()
      for (i in 1:nrow(std_rows)) {
        cat(sprintf(" %-15s %12.5f %12.5f %12.5f %12.5f\n",
                    "Std Dev", 
                    std_rows$Estimate[i],
                    std_rows$SE[i],
                    std_rows$`Z-score`[i],
                    std_rows$`P-value`[i]))
      }
      cat("\n")
    }
    cat("\n")
  }
  
  n_obs <- x$data_summary$n_obs
  n_sub <- x$data_summary$n_subjects
  
  # Print Stage 1
  if (!is.null(x$stage1)) {
    print_stage_sas_style(x$stage1, "Model without Scale Parameters", n_obs, n_sub)
  }
  
  # Print Stage 2
  if (!is.null(x$stage2)) {
    print_stage_sas_style(x$stage2, "Model With Scale Parameters", n_obs, n_sub)
  }
  
  # Print Stage 3
  s3 <- if (!is.null(x$stage3_best)) x$stage3_best else x$stage3
  
  if (!is.null(s3)) {
    m_type <- if (!is.null(s3$model_type)) s3$model_type else "unknown"
    print_stage_sas_style(s3, "Model With Random Scale", n_obs, n_sub, model_type = m_type)
  }
}