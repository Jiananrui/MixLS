// src/loglik.cpp

#include <omp.h>
#include <RcppArmadillo.h>
#include <limits>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate log-likelihood for Stage 1 & 2
//'
//' Computes the log-likelihood for the Stage 1 & 2 models using fixed
//' Gaussian-Hermite quadrature.
//'
//' @param params Vector of parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Vector of response values.
//' @param id Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @return The total log-likelihood value for all subjects.
// [[Rcpp::export]]
double stage12_loglik_rcpp(const arma::vec& params,
                         const arma::mat& X,
                         const arma::mat& U,
                         const arma::mat& Z,
                         const arma::vec& y,
                         const arma::uvec& id,
                         const arma::vec& points,
                         const arma::vec& weights) {
    using namespace arma;
    
    // Parameter & Data extraction
    int n_fixed = X.n_cols;
    int n_random = U.n_cols;
    int nq = points.n_elem;
    
    vec beta = params.subvec(0, n_fixed - 1);
    vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    vec tau = params.subvec(n_fixed + n_random, params.n_elem - 1);
    
    // Initialize output and subject info
    double total_loglik = 0.0;
    uvec unique_ids = unique(id);
    int n_subjects = unique_ids.n_elem;

    // Parallel loop over subjects
    // The main calculation is parallelized. The `reduction(+:total_loglik)` clause
    // ensures that the log-likelihood contributions from each thread are safely
    // summed into the final total.
    #pragma omp parallel for reduction(+:total_loglik)
    for(int i = 0; i < n_subjects; i++) {
        // Get data for the current subject
        uvec idx = find(id == unique_ids(i));
        mat X_i = X.rows(idx);
        mat U_i = U.rows(idx);
        mat Z_i = Z.rows(idx);
        vec y_i = y.elem(idx);
        
        // Initialize accumulator for this subject's integrated likelihood
        double h_i = 0.0;

        // Pre-calculate subject-specific terms
        // These are calculated once per subject.
        vec U_alpha_i = U_i * alpha;
        vec Z_tau_i = Z_i * tau;
        vec X_beta_i = X_i * beta;
        vec sigma_v_sq_i = exp(U_alpha_i);
        vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        vec sigma_epsilon_sq_i = exp(Z_tau_i);

        // Numerical integration via quadrature
        for(int q = 0; q < nq; q++) {
            double theta_1i = points(q);
            double w = weights(q);
            
            // Calculate the likelihood of the data for the q-th quadrature point
            vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            vec epsilon = y_i - mu;
            
            vec f_i = log(2.0 * M_PI * sigma_epsilon_sq_i) + square(epsilon) / sigma_epsilon_sq_i;
            double l_i = exp(-0.5 * sum(f_i));

            // Sum the weighted likelihoods to approximate the integral
            h_i += l_i * w;
        }
        // Aggregate Log-Likelihood
        if (h_i > 0) {
            total_loglik += log(h_i);
        }
    }
    return total_loglik;
}


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate log-likelihood for Stage 1 & 2 with adaptive quadrature
//'
//' Computes the log-likelihood using adaptive Gaussian-Hermite quadrature, where
//' the integration grid is centered and scaled based on Empirical Bayes estimates.
//'
//' @param params Vector of parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Vector of response values.
//' @param id Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @param adaptive Integer flag (1 for adaptive, 0 for fixed).
//' @param theta_eb Vector of EB posterior means for each subject.
//' @param psd_eb Vector of EB posterior standard deviations for each subject.
//' @return The total log-likelihood value for all subjects.
// [[Rcpp::export]]
double stage12_loglik_adaptive(const arma::vec& params,
                             const arma::mat& X,
                             const arma::mat& U,
                             const arma::mat& Z,
                             const arma::vec& y,
                             const arma::uvec& id,
                             const arma::vec& points,
                             const arma::vec& weights,
                             int adaptive,
                             const arma::vec& theta_eb,
                             const arma::vec& psd_eb) {
    using namespace arma;
    
    // Dispatch to the non-adaptive version
    if (adaptive == 0) {
        return stage12_loglik_rcpp(params, X, U, Z, y, id, points, weights);
    }
    
    // Parameter & Data extraction
    int n_fixed = X.n_cols;
    int n_random = U.n_cols;
    int nq = points.n_elem;
    
    vec beta = params.subvec(0, n_fixed - 1);
    vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    vec tau = params.subvec(n_fixed + n_random, params.n_elem - 1);
    
    // Initialize output and subject info
    double total_loglik = 0.0;
    uvec unique_ids = unique(id);
    int n_subjects = unique_ids.n_elem;

    // Parallel loop over subjects
    #pragma omp parallel for reduction(+:total_loglik)
    for(int i = 0; i < n_subjects; i++) {
        uvec idx = find(id == unique_ids(i));
        mat X_i = X.rows(idx);
        mat U_i = U.rows(idx);
        mat Z_i = Z.rows(idx);
        vec y_i = y.elem(idx);
        
        double h_i = 0.0;
        
        // Pre-calculate subject-specific terms
        vec U_alpha_i = U_i * alpha;
        vec Z_tau_i = Z_i * tau;
        vec X_beta_i = X_i * beta;
        vec sigma_v_sq_i = exp(U_alpha_i);
        vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        vec sigma_epsilon_sq_i = exp(Z_tau_i);

        // Adaptive grid transformation
        vec current_points = points;
        vec current_weights = weights;
        
        if ((unsigned int)i < theta_eb.n_elem && (unsigned int)i < psd_eb.n_elem && psd_eb(i) > 1e-8) {
            current_points = theta_eb(i) + psd_eb(i) * points;
            current_weights = psd_eb(i) * exp(0.5*(points%points - current_points%current_points)) % weights;
        }

        // Numerical integration via quadrature
        for(int q = 0; q < nq; q++) {
            double theta_1i = current_points(q);
            double w = current_weights(q);
            
            vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            vec epsilon = y_i - mu;
            
            vec f_i = log(2.0 * M_PI * sigma_epsilon_sq_i) + square(epsilon) / sigma_epsilon_sq_i;
            double l_i = exp(-0.5 * sum(f_i));
            h_i += l_i * w;
        }
        // Aggregate log-likelihood
        total_loglik += log(std::max(h_i, std::numeric_limits<double>::min()));
    }
    return total_loglik;
}


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate log-likelihood for stage 3 mixed-effects location scale model
//'
//' Computes the log-likelihood for the stage 3 model using 2D fixed quadrature.
//'
//' @param params Vector of parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Vector of response values.
//' @param id_numeric Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @param model_type String specifying the Stage 3 model type.
//' @return The total log-likelihood value.
// [[Rcpp::export]]
double stage3_loglik_combined(const arma::vec& params,
                            const arma::mat& X,
                            const arma::mat& U,
                            const arma::mat& Z,
                            const arma::vec& y,
                            const arma::uvec& id_numeric,
                            const arma::vec& points,
                            const arma::vec& weights,
                            const std::string& model_type = "linear") {
    
    using namespace arma;
    
    // Parameter & Data extraction
    int n_fixed = X.n_cols;
    int n_random = U.n_cols;
    int n_ws = Z.n_cols;
    
    const vec beta = params.subvec(0, n_fixed - 1);
    const vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    const vec tau = params.subvec(n_fixed + n_random, n_fixed + n_random + n_ws - 1);
    const vec sigma = params.subvec(n_fixed + n_random + n_ws, params.n_elem - 1);
    
    // Initialize output and subject info
    const uvec unique_ids = unique(id_numeric);
    const int n_subjects = unique_ids.n_elem;
    const int nq = points.n_elem;
    double total_loglik = 0.0;
    
    // Parallel loop over subjects
    #pragma omp parallel for reduction(+:total_loglik)
    for (int i = 0; i < n_subjects; i++) {
        // Get subject-specific data
        const uvec idx = find(id_numeric == unique_ids(i));
        const vec y_i = y.elem(idx);
        const mat X_i = X.rows(idx);
        const mat U_i = U.rows(idx);
        const mat Z_i = Z.rows(idx);
        
        // Pre-calculate subject-specific terms
        const vec X_beta_i = X_i * beta;
        const vec Z_tau_i = Z_i * tau;
        const vec U_alpha_i = U_i * alpha;
        const vec sigma_v_sq_i = exp(U_alpha_i);
        const vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        
        // Initialize accumulator for this subject's integrated likelihood
        double h_i = 0.0;
        
        // Pre-allocate thread private vectors
        vec theta;
        if (model_type == "independent") { theta = vec(1); }
        else if (model_type == "linear") { theta = vec(2); }
        else if (model_type == "interaction" || model_type == "quadratic") { theta = vec(3); }
        else { throw std::runtime_error("Invalid model_type specified"); }
        
        // 2D numerical integration via quadrature
        for (int q1 = 0; q1 < nq; q1++) {
            const double theta_1i = points(q1);
            const double w1 = weights(q1);
            
            // Pre-compute terms that only depend on the outer loop (q1)
            const vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            const vec epsilon = y_i - mu;
            
            for (int q2 = 0; q2 < nq; q2++) {
                const double theta_2i = points(q2);
                const double w = w1 * weights(q2);
                
                // Construct the random effects vector for the variance model
                if (model_type == "independent") { theta(0) = theta_2i; }
                else if (model_type == "linear") { theta(0) = theta_1i; theta(1) = theta_2i; }
                else if (model_type == "interaction") { theta(0) = theta_1i; theta(1) = theta_1i * theta_2i; theta(2) = theta_2i; }
                else if (model_type == "quadratic") { theta(0) = theta_1i; theta(1) = theta_1i * theta_1i; theta(2) = theta_2i; }
                
                const vec sigma_epsilon_sq = exp(Z_tau_i + dot(sigma, theta));
                const vec f_i = log(2.0 * M_PI * sigma_epsilon_sq) + square(epsilon) / sigma_epsilon_sq;
                const double l_i = exp(-0.5 * sum(f_i));
                h_i += l_i * w;
            }
        }
        // Aggregate log-likelihood
        if (h_i > 0) {
            total_loglik += log(h_i);
        }
    }
    return total_loglik;
}


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate log-likelihood for Stage 3 with adaptive quadrature
//'
//' Computes the log-likelihood for the Stage 3 model using 2D adaptive quadrature.
//'
//' @param params Vector of parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Vector of response values.
//' @param id_numeric Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @param model_type String specifying the Stage 3 model type.
//' @param adaptive Integer flag (1 for adaptive, 0 for fixed).
//' @param eb_theta2 Matrix (n_subjects x 2) of EB posterior means.
//' @param eb_thetav Matrix (n_subjects x 3) of EB posterior VCV elements.
//' @return The total log-likelihood value.
// [[Rcpp::export]]
double stage3_loglik_adaptive_combined(const arma::vec& params,
                                      const arma::mat& X,
                                      const arma::mat& U,
                                      const arma::mat& Z,
                                      const arma::vec& y,
                                      const arma::uvec& id_numeric,
                                      const arma::vec& points,
                                      const arma::vec& weights,
                                      const std::string& model_type,
                                      int adaptive,
                                      const arma::mat& eb_theta2,
                                      const arma::mat& eb_thetav) {
    using namespace arma;
    
    // Dispatch to non-adaptive version
    if (adaptive == 0) {
        return stage3_loglik_combined(params, X, U, Z, y, id_numeric, points, weights, model_type);
    }
    
    // Parameter & Data extraction
    int n_fixed = X.n_cols;
    int n_random = U.n_cols;
    int n_ws = Z.n_cols;
    const vec beta = params.subvec(0, n_fixed - 1);
    const vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    const vec tau = params.subvec(n_fixed + n_random, n_fixed + n_random + n_ws - 1);
    const vec sigma = params.subvec(n_fixed + n_random + n_ws, params.n_elem - 1);
    
    // Initialize output and subject info
    const uvec unique_ids = unique(id_numeric);
    const int n_subjects = unique_ids.n_elem;
    const int nq = points.n_elem;
    const double log_2pi = std::log(2.0 * M_PI);
    double total_loglik = 0.0;

    // Parallel loop over subjects
    #pragma omp parallel for reduction(+:total_loglik)
    for (int i = 0; i < n_subjects; i++) {
        // --- Get subject-specific data ---
        const uvec idx = find(id_numeric == unique_ids(i));
        const vec y_i = y.elem(idx);
        const mat X_i = X.rows(idx);
        const mat U_i = U.rows(idx);
        const mat Z_i = Z.rows(idx);
        
        // Pre-calculate subject-specific terms
        const vec X_beta_i = X_i * beta;
        const vec Z_tau_i = Z_i * tau;
        const vec U_alpha_i = U_i * alpha;
        const vec sigma_v_sq_i = exp(U_alpha_i);
        const vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        
        // Initialize accumulator for this subject
        double h_i = 0.0;

        // Adaptive grid transformation
        vec current_points_1 = points, current_weights_1 = weights;
        vec current_points_2 = points, current_weights_2 = weights;
        if (static_cast<unsigned int>(i) < eb_theta2.n_rows) {
            double mean1 = eb_theta2(i, 0);
            double mean2 = eb_theta2(i, 1);
            double var1 = eb_thetav(i, 0);
            double var2 = eb_thetav(i, 2);

            if (var1 > 1e-8) {
                double sd1 = std::sqrt(var1);
                current_points_1 = mean1 + sd1 * points;
                current_weights_1 = sd1 * exp(0.5 * (points % points - square(current_points_1))) % weights;
            }
            if (var2 > 1e-8) {
                double sd2 = std::sqrt(var2);
                current_points_2 = mean2 + sd2 * points;
                current_weights_2 = sd2 * exp(0.5 * (points % points - square(current_points_2))) % weights;
            }
        }
        
        // Pre-allocate thread-private vectors
        vec theta;
        if (model_type == "independent") { theta = vec(1); }
        else if (model_type == "linear") { theta = vec(2); }
        else if (model_type == "interaction" || model_type == "quadratic") { theta = vec(3); }

        // Numerical integration via 2D adaptive quadrature
        for (int q1 = 0; q1 < nq; q1++) {
            const double theta_1i = current_points_1(q1);
            const double w1 = current_weights_1(q1);
            
            // Pre-compute terms depending on q1
            const vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            const vec epsilon = y_i - mu;
            
            for (int q2 = 0; q2 < nq; q2++) {
                const double theta_2i = current_points_2(q2);
                const double w = w1 * current_weights_2(q2);
                
                // Construct theta vector based on model type
                if (model_type == "independent") { theta(0) = theta_2i; }
                else if (model_type == "linear") { theta(0) = theta_1i; theta(1) = theta_2i; }
                else if (model_type == "interaction") { theta(0) = theta_1i; theta(1) = theta_1i * theta_2i; theta(2) = theta_2i; }
                else { theta(0) = theta_1i; theta(1) = theta_1i * theta_1i; theta(2) = theta_2i; }
                
                // Innermost loop calculations
                const vec sigma_epsilon_sq = exp(Z_tau_i + dot(sigma, theta));
                const vec f_i = log_2pi + log(sigma_epsilon_sq) + square(epsilon) / sigma_epsilon_sq;
                const double l_i = exp(-0.5 * sum(f_i));
                h_i += l_i * w;
            }
        }
        // Aggregate log-likelihood
        total_loglik += log(std::max(h_i, std::numeric_limits<double>::min()));
    }
    return total_loglik;
}