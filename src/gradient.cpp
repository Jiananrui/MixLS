// src/gradient.cpp

#include <omp.h>
#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate gradient for Stage 1 & 2 log-likelihood
//'
//' This function computes the gradient of the log-likelihood for the Stage 1 & 2
//' models using fixed Gaussian-Hermite quadrature.
//'
//' @param params Vector of parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Vector of response values.
//' @param id Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @return A vector containing the gradient of the log-likelihood.
// [[Rcpp::export]]
arma::vec stage12_gradient_rcpp(const arma::vec& params,
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
    int n_params = params.n_elem;
    int nq = points.n_elem;
    
    vec beta = params.subvec(0, n_fixed - 1);
    vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    vec tau = params.subvec(n_fixed + n_random, n_params - 1);
    
    // Initialize output vector
    vec gradient = zeros(n_params);
    
    // Get unique IDs
    uvec unique_ids = unique(id);
    int n_subjects = unique_ids.n_elem;
    
    // Parallel loop over subjects
    // The main computation is parallelized across subjects.
    #pragma omp parallel for
    for(int i = 0; i < n_subjects; i++) {
        // Thread private variables
        // Each thread calculates its own gradient contribution to avoid race conditions.
        vec gradient_private = zeros(n_params);
        uvec idx = find(id == unique_ids(i));
        
        // Extract subject data
        mat X_i = X.rows(idx);
        mat U_i = U.rows(idx);
        mat Z_i = Z.rows(idx);
        vec y_i = y.elem(idx);
        
        // Initialize accumulators for the subject's integrated likelihood and score
        // Denominator for the gradient calculation
        double h_i = 0.0;
        // Numerator for the gradient calculation
        vec d_i = zeros(n_params);
        
        // Pre-calculate subject-specific terms
        // These only depend on the subject, not the quadrature points
        vec U_alpha_i = U_i * alpha;
        vec Z_tau_i = Z_i * tau;
        vec X_beta_i = X_i * beta;
        vec sigma_v_sq_i = exp(U_alpha_i);
        vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        vec sigma_epsilon_sq_i = exp(Z_tau_i);
        vec sigma_epsilon_sq_inv_i = 1.0 / sigma_epsilon_sq_i;
        
        // Numerical integration via quadrature
        for(int q = 0; q < nq; q++) {
            double theta_1i = points(q);
            double w = weights(q);
            
            // Calculate terms that depend on the quadrature point
            vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            vec epsilon = y_i - mu;
            
            vec epsilon_sigma_ratio = epsilon % sigma_epsilon_sq_inv_i;
            vec epsilon_sq_sigma_ratio = square(epsilon) % sigma_epsilon_sq_inv_i;
            
            // Compute the gradient of log-likelihood for one observation for each parameter
            vec grad_beta = -2.0 * X_i.t() * epsilon_sigma_ratio;
            vec grad_alpha = -U_i.t() * (epsilon_sigma_ratio * theta_1i % sqrt_sigma_v_sq_i);
            vec grad_tau = Z_i.t() * (1.0 - epsilon_sq_sigma_ratio);
            
            // Combine gradient components
            vec grad_li = join_vert(grad_beta, grad_alpha, grad_tau);
            grad_li *= -0.5;
            
            // Compute the likelihood value at this quadrature point
            const double log_2pi = std::log(2.0 * M_PI);
            vec f_i = log_2pi + log(sigma_epsilon_sq_i) + epsilon_sq_sigma_ratio;
            double l_i = exp(-0.5 * sum(f_i));
            
            // Update the integrated likelihood (h_i) and the integrated score (d_i)
            h_i += l_i * w;
            d_i += grad_li * l_i * w;
        }
        
        // Calculate contribution from this subject and store in private vector
        // The gradient for subject i is E[score|y_i] = d_i / h_i
        if (h_i > 1e-100) {
            gradient_private += d_i / h_i;
        }
        // Safely aggregate result
        #pragma omp critical
        {
            gradient += gradient_private;
        }
    }
    return gradient;
}


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate gradient for Stage 1 & 2 with adaptive quadrature
//'
//' This function computes the gradient using adaptive Gaussian-Hermite quadrature,
//' where the integration grid is centered and scaled based on Empirical Bayes
//' estimates from the previous iteration.
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
//' @return A vector containing the gradient of the log-likelihood.
// [[Rcpp::export]]
arma::vec stage12_gradient_adaptive(const arma::vec& params,
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
    
    // Parameter & Data extraction
    int n_fixed = X.n_cols;
    int n_random = U.n_cols;
    int n_params = params.n_elem;
    int nq = points.n_elem;
    
    vec beta = params.subvec(0, n_fixed - 1);
    vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    vec tau = params.subvec(n_fixed + n_random, n_params - 1);
    
    // Initialize output vector
    vec gradient = zeros(n_params);
    uvec unique_ids = unique(id);
    int n_subjects = unique_ids.n_elem;

    // Parallel loop over subjects
    #pragma omp parallel for
    for(int i = 0; i < n_subjects; i++) {
        // Thread private variables
        vec gradient_private = zeros(n_params);
        uvec idx = find(id == unique_ids(i));
        
        // Extract subject data
        mat X_i = X.rows(idx);
        mat U_i = U.rows(idx);
        mat Z_i = Z.rows(idx);
        vec y_i = y.elem(idx);
        
        // Initialize accumulators
        double h_i = 0.0;
        vec d_i = zeros(n_params);
        
        // Pre-calculate subject-specific Terms
        vec U_alpha_i = U_i * alpha;
        vec Z_tau_i = Z_i * tau;
        vec X_beta_i = X_i * beta;
        vec sigma_v_sq_i = exp(U_alpha_i);
        vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        vec sigma_epsilon_sq_i = exp(Z_tau_i);
        vec sigma_epsilon_sq_inv_i = 1.0 / sigma_epsilon_sq_i;
        
        // Adaptive grid transformation
        vec current_points = points;
        vec current_weights = weights;
        
        if (adaptive == 1 && (unsigned int)i < theta_eb.n_elem && (unsigned int)i < psd_eb.n_elem) {
            current_points = theta_eb(i) + psd_eb(i) * points;
            current_weights = psd_eb(i) * exp(0.5*(points%points - current_points%current_points)) % weights;
        }
        
        // Numerical integration via quadrature
        for(int q = 0; q < nq; q++) {
            double theta_1i = current_points(q);
            double w = current_weights(q);
            
            // Calculations depending on quadrature point
            vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            vec epsilon = y_i - mu;
            
            vec epsilon_sigma_ratio = epsilon % sigma_epsilon_sq_inv_i;
            vec epsilon_sq_sigma_ratio = square(epsilon) % sigma_epsilon_sq_inv_i;
            
            // Compute gradient components
            vec grad_beta = -2.0 * X_i.t() * epsilon_sigma_ratio;
            vec grad_alpha = -U_i.t() * (epsilon_sigma_ratio * theta_1i % sqrt_sigma_v_sq_i);
            vec grad_tau = Z_i.t() * (1.0 - epsilon_sq_sigma_ratio);
            
            // Combine gradient components
            vec grad_li = join_vert(grad_beta, grad_alpha, grad_tau);
            grad_li *= -0.5;
            
            // Compute likelihood term
            const double log_2pi = std::log(2.0 * M_PI);
            vec f_i = log_2pi + log(sigma_epsilon_sq_i) + epsilon_sq_sigma_ratio;
            double l_i = exp(-0.5 * sum(f_i));
            
            // Update accumulators
            h_i += l_i * w;
            d_i += grad_li * l_i * w;
        }
        // Calculate and aggregate gradient contribution
        if (h_i > 1e-100) {
            gradient_private += d_i / h_i;
        }
        #pragma omp critical
        {
            gradient += gradient_private;
        }
    }
    return gradient;
}


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate gradient for Stage 3 mixed-effects location-scale model
//'
//' This function computes the gradient for the Stage 3 model, which includes
//' bivariate random effects for location and scale. It uses 2D numerical
//' integration and is parallelized across subjects.
//'
//' @param params Vector of parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Vector of response values.
//' @param id Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @param model_type A string specifying the Stage 3 model type ("linear", etc.).
//' @return A vector containing the gradient of the log-likelihood.
// [[Rcpp::export]]
arma::vec stage3_gradient_combined(const arma::vec& params,
                                const arma::mat& X,
                                const arma::mat& U,
                                const arma::mat& Z,
                                const arma::vec& y,
                                const arma::uvec& id,
                                const arma::vec& points,
                                const arma::vec& weights,
                                const std::string& model_type = "linear") {
    using namespace arma;
    
    // Parameter & Data extraction
    const int n_fixed = X.n_cols;
    const int n_random = U.n_cols;
    const int n_bs = Z.n_cols;
    const int n_params = params.n_elem;
    const int nq = points.n_elem;
    const double log_2pi = std::log(2.0 * M_PI);
    
    const vec beta = params.subvec(0, n_fixed - 1);
    const vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    const vec tau = params.subvec(n_fixed + n_random, n_fixed + n_random + n_bs - 1);
    const vec sigma = params.subvec(n_fixed + n_random + n_bs, n_params - 1);
    
    // Initialize output vector
    vec gradient = zeros(n_params);
    const uvec unique_ids = unique(id);
    const int n_subjects = unique_ids.n_elem;
    
    // Parallel loop over subjects
    #pragma omp parallel for
    for(int i = 0; i < n_subjects; i++) {
        // Thread private variables
        vec gradient_private = zeros(n_params);
        const uvec idx = find(id == unique_ids(i));
        
        // Extract subject data and perform pre-calculations
        const mat X_i = X.rows(idx);
        const mat U_i = U.rows(idx);
        const mat Z_i = Z.rows(idx);
        const vec y_i = y.elem(idx);
        const vec X_beta_i = X_i * beta;
        const vec Z_tau_i = Z_i * tau;
        const vec U_alpha_i = U_i * alpha;
        const vec sigma_v_sq_i = exp(U_alpha_i);
        const vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);

        // Initialize accumulators for the subject
        double h_i = 0.0;
        vec d_i = zeros(n_params);
        
        // Pre-allocate the theta vector for the variance model
        vec theta;
        if (model_type == "independent") { theta = vec(1); }
        else if (model_type == "linear") { theta = vec(2); }
        else if (model_type == "interaction" || model_type == "quadratic") { theta = vec(3); }
        else { throw std::runtime_error("Invalid model_type specified"); }
        
        // 2D numerical integration via quadrature
        for(int q1 = 0; q1 < nq; q1++) {
            const double theta_1i = points(q1);
            const double w1 = weights(q1);
            
            // Pre-compute terms that only depend on the outer quadrature loop (q1)
            const vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            vec epsilon = y_i - mu;
            
            for(int q2 = 0; q2 < nq; q2++) {
                const double theta_2i = points(q2);
                const double w = w1 * weights(q2);
                
                // Construct the random effects vector for the variance model
                if (model_type == "independent") { theta(0) = theta_2i; }
                else if (model_type == "linear") { theta(0) = theta_1i; theta(1) = theta_2i; }
                else if (model_type == "interaction") { theta(0) = theta_1i; theta(1) = theta_1i * theta_2i; theta(2) = theta_2i; }
                else if (model_type == "quadratic") { theta(0) = theta_1i; theta(1) = theta_1i * theta_1i; theta(2) = theta_2i; }
                
                // Calculate variance and common ratios
                const double sigma_theta_val = dot(sigma, theta);
                vec sigma_epsilon_sq = exp(Z_tau_i + sigma_theta_val);
                vec sigma_epsilon_sq_inv = 1.0 / sigma_epsilon_sq;
                
                vec epsilon_sigma_ratio = epsilon % sigma_epsilon_sq_inv;
                vec epsilon_sq_sigma_ratio = square(epsilon) % sigma_epsilon_sq_inv;
                
                // Compute gradient components for each parameter block
                const vec grad_beta = -2.0 * X_i.t() * epsilon_sigma_ratio;
                const vec grad_alpha = -U_i.t() * (epsilon_sigma_ratio * theta_1i % sqrt_sigma_v_sq_i);
                const vec grad_tau = Z_i.t() * (1.0 - epsilon_sq_sigma_ratio);
                
                vec grad_sigma = zeros(theta.n_elem);
                const double sum_ones_minus_ratio = sum(1.0 - epsilon_sq_sigma_ratio);
                if (model_type == "independent") { grad_sigma(0) = sum_ones_minus_ratio * theta_2i; }
                else if (model_type == "linear") { grad_sigma(0) = sum_ones_minus_ratio * theta_1i; grad_sigma(1) = sum_ones_minus_ratio * theta_2i; }
                else if (model_type == "interaction") { grad_sigma(0) = sum_ones_minus_ratio * theta_1i; grad_sigma(1) = sum_ones_minus_ratio * (theta_1i * theta_2i); grad_sigma(2) = sum_ones_minus_ratio * theta_2i; }
                else if (model_type == "quadratic") { grad_sigma(0) = sum_ones_minus_ratio * theta_1i; grad_sigma(1) = sum_ones_minus_ratio * (theta_1i * theta_1i); grad_sigma(2) = sum_ones_minus_ratio * theta_2i; }
                
                // Assemble the full score vector
                vec grad_li(n_params);
                grad_li.subvec(0, n_fixed - 1) = grad_beta;
                grad_li.subvec(n_fixed, n_fixed + n_random - 1) = grad_alpha;
                grad_li.subvec(n_fixed + n_random, n_fixed + n_random + n_bs - 1) = grad_tau;
                grad_li.subvec(n_fixed + n_random + n_bs, n_params - 1) = grad_sigma;
                grad_li *= -0.5;
                
                // Calculate likelihood at this grid point
                const vec f_i = log_2pi + log(sigma_epsilon_sq) + epsilon_sq_sigma_ratio;
                const double l_i = exp(-0.5 * sum(f_i));
                
                // Update accumulators
                h_i += l_i * w;
                d_i += grad_li * l_i * w;
            }
        }
        
        // Calculate and aggregate gradient contribution
        if (h_i > 1e-100) {
            gradient_private += d_i / h_i;
        }
        #pragma omp critical
        {
            gradient += gradient_private;
        }
    }
    return gradient;
}


// ------------------------------------------------------------------------------------------------------------------------------


//' Calculate gradient for Stage 3 with adaptive quadrature
//'
//' This function computes the gradient for the Stage 3 model using 2D adaptive
//' Gaussian-Hermite quadrature.
//'
//' @param params Vector of parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Vector of response values.
//' @param id_numeric Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @param model_type A string specifying the Stage 3 model type.
//' @param adaptive Integer flag (1 for adaptive, 0 for fixed).
//' @param eb_theta2 Matrix (n_subjects x 2) of EB posterior means.
//' @param eb_thetav Matrix (n_subjects x 3) of EB posterior VCV elements.
//' @return A vector containing the gradient of the log-likelihood.
// [[Rcpp::export]]
arma::vec stage3_gradient_adaptive_combined(const arma::vec& params,
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

    // This function acts as a dispatcher.
    if (adaptive == 0) {
        return stage3_gradient_combined(params, X, U, Z, y, id_numeric, points, weights, model_type);
    }
    
    // Parameter & Data extraction
    const int n_fixed = X.n_cols;
    const int n_random = U.n_cols;
    const int n_bs = Z.n_cols;
    const int n_params = params.n_elem;
    const int nq = points.n_elem;
    
    const vec beta = params.subvec(0, n_fixed - 1);
    const vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    const vec tau = params.subvec(n_fixed + n_random, n_fixed + n_random + n_bs - 1);
    const vec sigma = params.subvec(n_fixed + n_random + n_bs, n_params - 1);
    
    // Initialize output vector
    vec gradient = zeros(n_params);
    const uvec unique_ids = unique(id_numeric);
    const int n_subjects = unique_ids.n_elem;
    
    // Parallel loop over subjects
    #pragma omp parallel for
    for(int i = 0; i < n_subjects; i++) {
        // Thread private variables
        vec gradient_private = zeros(n_params);
        const uvec idx = find(id_numeric == unique_ids(i));
        
        // Extract subject data and perform pre-calculations
        const mat X_i = X.rows(idx);
        const mat U_i = U.rows(idx);
        const mat Z_i = Z.rows(idx);
        const vec y_i = y.elem(idx);
        const vec X_beta_i = X_i * beta;
        const vec Z_tau_i = Z_i * tau;
        const vec U_alpha_i = U_i * alpha;
        const vec sigma_v_sq_i = exp(U_alpha_i);
        const vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        
        // Initialize accumulators
        double h_i = 0.0;
        vec d_i = zeros(n_params);
        
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
        
        // Pre-allocate theta vectors
        vec theta;
        if (model_type == "independent") { theta = vec(1); }
        else if (model_type == "linear") { theta = vec(2); }
        else if (model_type == "interaction" || model_type == "quadratic") { theta = vec(3); }

        // 2D numerical integration via quadrature
        for(int q1 = 0; q1 < nq; q1++) {
            const double theta_1i = current_points_1(q1);
            const double w1 = current_weights_1(q1);
            
            // Pre-compute terms depending on q1
            const vec mu = X_beta_i + sqrt_sigma_v_sq_i * theta_1i;
            const vec epsilon = y_i - mu;
            
            for(int q2 = 0; q2 < nq; q2++) {
                const double theta_2i = current_points_2(q2);
                const double w = w1 * current_weights_2(q2);
                
                // Calculations depending on q1 and q2
                if (model_type == "independent") { theta(0) = theta_2i; }
                else if (model_type == "linear") { theta(0) = theta_1i; theta(1) = theta_2i; }
                else if (model_type == "interaction") { theta(0) = theta_1i; theta(1) = theta_1i * theta_2i; theta(2) = theta_2i; }
                else { theta(0) = theta_1i; theta(1) = theta_1i * theta_1i; theta(2) = theta_2i; }
                
                // Calculate variance and common ratios
                const vec sigma_epsilon_sq = exp(Z_tau_i + dot(sigma, theta));
                const vec sigma_epsilon_sq_inv = 1.0 / sigma_epsilon_sq;
                const vec epsilon_sigma_ratio = epsilon % sigma_epsilon_sq_inv;
                const vec epsilon_sq_sigma_ratio = square(epsilon) % sigma_epsilon_sq_inv;
                
                // Compute gradient components for each parameter block
                vec grad_beta = -2.0 * X_i.t() * epsilon_sigma_ratio;
                vec grad_alpha = -U_i.t() * (epsilon_sigma_ratio * theta_1i % sqrt_sigma_v_sq_i);
                vec grad_tau = Z_i.t() * (1.0 - epsilon_sq_sigma_ratio);
                
                vec grad_sigma = zeros(theta.n_elem);
                double sum_ones_minus_ratio = sum(1.0 - epsilon_sq_sigma_ratio);
                for(unsigned int k=0; k < theta.n_elem; ++k) {
                    grad_sigma(k) = sum_ones_minus_ratio * theta(k);
                }
                
                // Assemble the full score vector
                vec grad_li(n_params);
                grad_li.subvec(0, n_fixed - 1) = grad_beta;
                grad_li.subvec(n_fixed, n_fixed + n_random - 1) = grad_alpha;
                grad_li.subvec(n_fixed + n_random, n_fixed + n_random + n_bs - 1) = grad_tau;
                grad_li.subvec(n_fixed + n_random + n_bs, n_params - 1) = grad_sigma;
                grad_li *= -0.5;
                
                // Calculate likelihood at this grid point
                const vec f_i = log(2.0 * M_PI * sigma_epsilon_sq) + epsilon_sq_sigma_ratio;
                const double l_i = exp(-0.5 * sum(f_i));
                
                // Update accumulators
                h_i += l_i * w;
                d_i += grad_li * l_i * w;
            }
        }
        // Calculate and Aggregate Gradient Contribution
        if (h_i > 1e-100) {
            gradient_private += d_i / h_i;
        }
        #pragma omp critical
        {
            gradient += gradient_private;
        }
    }
    return gradient;
}
