// src/eb_estimates.cpp

#include <omp.h>
#include <RcppArmadillo.h>
#include <limits>
// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::plugins(openmp)]]

// ------------------------------------------------------------------------------------------------------------------------------


//' Update Empirical Bayes Estimates for Stage 1 & 2
//'
//' This function calculates the posterior mean and standard deviation for the random
//' intercept in the Stage 1 & 2 models. It uses the parameter estimates from the
//' current iteration to perform numerical integration via Gaussian quadrature for
//' each subject. The calculation is parallelized across subjects using OpenMP.
//'
//' @param params Vector of current model parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Response vector.
//' @param id Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @param eb_prev_theta EB mean estimates from the previous iteration.
//' @param eb_prev_psd EB standard deviation estimates from the previous iteration.
//' @return A Rcpp::List containing two named elements: `theta_eb` (a vector of the
//'   updated posterior means) and `psd_eb` (a vector of the updated posterior
//'   standard deviations).
// [[Rcpp::export]]
Rcpp::List update_eb_estimates_rcpp(const arma::vec& params,
                                    const arma::mat& X, const arma::mat& U, const arma::mat& Z,
                                    const arma::vec& y, const arma::uvec& id,
                                    const arma::vec& points, const arma::vec& weights,
                                    const arma::vec& eb_prev_theta, const arma::vec& eb_prev_psd) {
    using namespace arma;

    // Parameter & Data extraction
    int n_fixed = X.n_cols;
    int n_random = U.n_cols;
    vec beta = params.subvec(0, n_fixed - 1);
    vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
    vec tau = params.subvec(n_fixed + n_random, params.n_elem - 1);

    uvec unique_ids = unique(id);
    uword n_subjects = unique_ids.n_elem; // Use uword for size
    int nq = points.n_elem;

    // Initialize output vectors
    vec theta_new = zeros<vec>(n_subjects);
    vec psd_new = zeros<vec>(n_subjects);

    // Parallel loop over subjects
    #pragma omp parallel for
    for (uword i = 0; i < n_subjects; ++i) {
        // Get data for the current subject
        uvec idx = find(id == unique_ids(i));
        mat X_i = X.rows(idx);
        mat U_i = U.rows(idx);
        mat Z_i = Z.rows(idx);
        vec y_i = y.elem(idx);

        // Adaptive grid transformation
        // If using adaptive quadrature, transform the standard points and weights
        // to be centered and scaled by the previous iteration's EB estimates.
        vec current_points = points;
        vec current_weights = weights;

        if (i < eb_prev_theta.n_elem && i < eb_prev_psd.n_elem && eb_prev_psd(i) > 1e-8) {
            current_points = eb_prev_theta(i) + eb_prev_psd(i) * points;
            current_weights = weights * eb_prev_psd(i) % exp(0.5 * (square(points) - square(current_points)));
        }

        // Pre-calculate subject-specific terms
        // These are calculated once per subject to avoid redundant work inside the quadrature loop.
        vec U_alpha_i = U_i * alpha;
        vec Z_tau_i = Z_i * tau;
        vec X_beta_i = X_i * beta;
        vec sigma_v_sq_i = exp(U_alpha_i);
        vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
        vec sigma_epsilon_sq_i = exp(Z_tau_i);

        // Numerical integration via quadrature
        double h_i = 0.0;
        vec likelihood_values = zeros<vec>(nq);

        for (int q = 0; q < nq; ++q) {
            // Calculate the likelihood of the data for the q-th quadrature point
            vec mu = X_beta_i + sqrt_sigma_v_sq_i * current_points(q);
            vec epsilon = y_i - mu;
            double log_lik_sum = sum(log(2 * M_PI * sigma_epsilon_sq_i) + (square(epsilon) / sigma_epsilon_sq_i));
            likelihood_values(q) = exp(std::max(-0.5 * log_lik_sum, -700.0));
            h_i += likelihood_values(q) * current_weights(q);
        }
        
        // Compute new EB estimates
        // Calculate the posterior mean and variance using the results from the integration.
        if (h_i > std::numeric_limits<double>::min()) {
            // Normalizing constant
            double scal = 1.0 / h_i;
            // Posterior mean: E[theta|y] = integral(theta * L(theta|y)) d(theta) / integral(L(theta|y)) d(theta)
            double numerator_theta = sum(likelihood_values % current_points % current_weights);
            theta_new(i) = scal * numerator_theta;

            // Posterior variance: Var(theta|y) = E[(theta - E[theta|y])^2 | y]
            double numerator_var = sum(likelihood_values % square(current_points - theta_new(i)) % current_weights);
            psd_new(i) = sqrt(std::max(scal * numerator_var, 1e-12));
        } else {
            // If integration fails (h_i is too small), fall back to the previous estimate.
            theta_new(i) = eb_prev_theta(i);
            psd_new(i) = eb_prev_psd(i);
        }
    }

    return Rcpp::List::create(Rcpp::Named("theta_eb") = theta_new,
                                Rcpp::Named("psd_eb") = psd_new);
}


// ------------------------------------------------------------------------------------------------------------------------------


//' Update Bivariate Empirical Bayes Estimates for Stage 3
//'
//' This function calculates the posterior mean vector (2x1) and variance-covariance matrix
//' (2x2) for the bivariate random effects (location and scale) in the Stage 3 model.
//' It uses 2D numerical integration and is parallelized across subjects.
//'
//' @param params Vector of current model parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y Response vector.
//' @param id Vector of subject identifiers (0-based).
//' @param points Standard Gaussian-Hermite quadrature points.
//' @param weights Standard Gaussian-Hermite quadrature weights.
//' @param model_type A string specifying the Stage 3 model type ("linear", "quadratic", etc.).
//' @param adaptive An integer flag (1 for adaptive quadrature, 0 for fixed).
//' @param eb_prev_theta2 A matrix (n_subjects x 2) of the posterior means from the previous iteration.
//' @param eb_prev_thetav A matrix (n_subjects x 3) of the unique posterior
//'   variance-covariance elements [Var1, Cov, Var2] from the previous iteration.
//' @return A Rcpp::List containing two named elements: `theta2_eb` (a matrix of the
//'   updated posterior means) and `thetav_eb` (a matrix of the updated posterior
//'   variance-covariance elements).
// [[Rcpp::export]]
Rcpp::List update_eb_estimates_stage3_rcpp(const arma::vec& params,
                                           const arma::mat& X, const arma::mat& U, const arma::mat& Z,
                                           const arma::vec& y, const arma::uvec& id,
                                           const arma::vec& points, const arma::vec& weights,
                                           const std::string& model_type, int adaptive,
                                           const arma::mat& eb_prev_theta2, const arma::mat& eb_prev_thetav) {
    using namespace arma;

    // Parameter & Data extraction
    int n_fixed = X.n_cols;
    int n_random_bs = U.n_cols;
    int n_random_ws = Z.n_cols;
    vec beta = params.subvec(0, n_fixed - 1);
    vec alpha = params.subvec(n_fixed, n_fixed + n_random_bs - 1);
    vec tau = params.subvec(n_fixed + n_random_bs, n_fixed + n_random_bs + n_random_ws - 1);
    vec sigma = params.subvec(n_fixed + n_random_bs + n_random_ws, params.n_elem - 1);

    uvec unique_ids = unique(id);
    uword n_subjects = unique_ids.n_elem;
    int nq = points.n_elem;

    // Initialize output matrices
    mat theta2_new = zeros<mat>(n_subjects, 2);
    mat thetav_new = zeros<mat>(n_subjects, 3);

    // Parallel Loop Over Subjects
    #pragma omp parallel for
    for (uword i = 0; i < n_subjects; ++i) {
        // Get data for the current subject
        uvec idx = find(id == unique_ids(i));
        mat X_i = X.rows(idx);
        mat U_i = U.rows(idx);
        mat Z_i = Z.rows(idx);
        vec y_i = y.elem(idx);

        // Pre-calculate subject-specific terms
        vec X_beta_i = X_i * beta;
        vec Z_tau_i = Z_i * tau;
        vec bsvar_i = exp(U_i * alpha);

        // Adaptive grid transformation
        // Create separate adaptive grids for each of the two random effects.
        vec points1 = points, weights1 = weights;
        vec points2 = points, weights2 = weights;

        if (adaptive == 1 && i < eb_prev_theta2.n_rows) {
            if (eb_prev_thetav(i, 0) > 1e-8) { // Check variance of theta1
                double sd1 = sqrt(eb_prev_thetav(i, 0));
                points1 = eb_prev_theta2(i, 0) + sd1 * points;
                weights1 = weights * sd1 % exp(0.5 * (square(points) - square(points1)));
            }
            if (eb_prev_thetav(i, 2) > 1e-8) { // Check variance of theta2
                double sd2 = sqrt(eb_prev_thetav(i, 2));
                points2 = eb_prev_theta2(i, 1) + sd2 * points;
                weights2 = weights * sd2 % exp(0.5 * (square(points) - square(points2)));
            }
        }

        // 2D numerical integration via quadrature
        mat likelihood_values = zeros<mat>(nq, nq);
        double h_i = 0.0;
        
        // Pre-allocate the theta vector used in the variance model
        vec theta_vec;
        if (model_type == "independent") { theta_vec.set_size(1); }
        else if (model_type == "linear") { theta_vec.set_size(2); }
        else { theta_vec.set_size(3); }

        for (int q1 = 0; q1 < nq; ++q1) {
            double theta_1i = points1(q1);
            // Pre-calculate residual part that only depends on the outer loop
            vec errij = y_i - (X_beta_i + sqrt(bsvar_i) * theta_1i);

            for (int q2 = 0; q2 < nq; ++q2) {
                double theta_2i = points2(q2);
                
                // Construct the appropriate random effects vector for the variance model
                if (model_type == "independent") { theta_vec(0) = theta_2i; }
                else if (model_type == "linear") { theta_vec(0) = theta_1i; theta_vec(1) = theta_2i; }
                else if (model_type == "quadratic") { theta_vec(0) = theta_1i; theta_vec(1) = theta_1i*theta_1i; theta_vec(2) = theta_2i; }
                else { theta_vec(0) = theta_1i; theta_vec(1) = theta_1i*theta_2i; theta_vec(2) = theta_2i; }
                
                // Calculate likelihood at this grid point
                vec wsvar = exp(Z_tau_i + as_scalar(theta_vec.t() * sigma));
                double log_prob_sum = sum(log(2 * M_PI * wsvar) + (square(errij) / wsvar));
                double lik_val = exp(std::max(-0.5 * log_prob_sum, -700.0));
                likelihood_values(q1, q2) = lik_val;

                // Add to the integrated likelihood
                h_i += lik_val * weights1(q1) * weights2(q2);
            }
        }

        // Compute new bivariate EB estimates
        if (h_i > 1e-100) {
            double scal = 1.0 / h_i;
            // Outer product of weights
            mat w_mat = weights1 * weights2.t(); 
            
            // Posterior Means: E[theta_k|y]
            double mean1_num = accu(likelihood_values % repmat(points1, 1, nq) % w_mat);
            double mean2_num = accu(likelihood_values % repmat(points2.t(), nq, 1) % w_mat);
            theta2_new(i, 0) = scal * mean1_num;
            theta2_new(i, 1) = scal * mean2_num;
            
            // Posterior variances & covariance: E[(theta_k - E[theta_k|y]) * (theta_j - E[theta_j|y]) | y]
            mat diff1_mat = repmat(points1 - theta2_new(i, 0), 1, nq);
            mat diff2_mat = repmat((points2 - theta2_new(i, 1)).t(), nq, 1);
            
            double var1_num = accu(likelihood_values % square(diff1_mat) % w_mat);
            double var2_num = accu(likelihood_values % square(diff2_mat) % w_mat);
            double cov_num  = accu(likelihood_values % (diff1_mat % diff2_mat) % w_mat);
            
            thetav_new(i, 0) = std::max(scal * var1_num, 1e-12); // Var(theta1)
            thetav_new(i, 1) = scal * cov_num;                   // Cov(theta1, theta2)
            thetav_new(i, 2) = std::max(scal * var2_num, 1e-12); // Var(theta2)
        } else {
            // Fallback to previous estimate if integration fails
            theta2_new.row(i) = eb_prev_theta2.row(i);
            thetav_new.row(i) = eb_prev_thetav.row(i);
        }
    }

    return Rcpp::List::create(Rcpp::Named("theta2_eb") = theta2_new,
                                Rcpp::Named("thetav_eb") = thetav_new);
}