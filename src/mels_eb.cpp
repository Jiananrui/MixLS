// src/mels_eb.cpp

#include <RcppArmadillo.h>
#include <RcppParallel.h>
#include <limits>

// [[Rcpp::depends(RcppArmadillo, RcppParallel)]]

using namespace arma;
using namespace RcppParallel;

// -----------------------------------------------------------------------------
// WORKER 1: Stage 1 & 2 EB Estimates
// -----------------------------------------------------------------------------

struct EBEstimatesWorker : public Worker {
    // Inputs
    const vec& params;
    const mat& X;
    const mat& U;
    const mat& Z;
    const vec& y;
    const uvec& id;
    const uvec& unique_ids;
    const vec& points;
    const vec& weights;
    const vec& eb_prev_theta;
    const vec& eb_prev_psd;

    // Outputs
    vec& theta_new;
    vec& psd_new;

    // Internal dimensions
    int n_fixed, n_random, nq;

    // Constructor
    EBEstimatesWorker(const vec& params, const mat& X, const mat& U, const mat& Z,
                      const vec& y, const uvec& id, const uvec& unique_ids,
                      const vec& points, const vec& weights,
                      const vec& eb_prev_theta, const vec& eb_prev_psd,
                      vec& theta_new, vec& psd_new)
        : params(params), X(X), U(U), Z(Z), y(y), id(id), unique_ids(unique_ids),
          points(points), weights(weights), eb_prev_theta(eb_prev_theta), 
          eb_prev_psd(eb_prev_psd), theta_new(theta_new), psd_new(psd_new)
    {
        n_fixed = X.n_cols;
        n_random = U.n_cols;
        nq = points.n_elem;
    }

    // The parallel loop
    void operator()(std::size_t begin, std::size_t end) {
        
        vec beta = params.subvec(0, n_fixed - 1);
        vec alpha = params.subvec(n_fixed, n_fixed + n_random - 1);
        vec tau = params.subvec(n_fixed + n_random, params.n_elem - 1);

        // Pre-allocate thread local buffers
        vec likelihood_values(nq);

        for (std::size_t i = begin; i < end; ++i) {
            uvec idx = find(id == unique_ids(i));
            mat X_i = X.rows(idx);
            mat U_i = U.rows(idx);
            mat Z_i = Z.rows(idx);
            vec y_i = y.elem(idx);

            // Adaptive grid setup
            vec current_points = points;
            vec current_weights = weights;

            if (i < eb_prev_theta.n_elem && i < eb_prev_psd.n_elem && eb_prev_psd(i) > 1e-8) {
                current_points = eb_prev_theta(i) + eb_prev_psd(i) * points;
                current_weights = weights * eb_prev_psd(i) % exp(0.5 * (square(points) - square(current_points)));
            }

            // Pre-calculate subject terms
            vec U_alpha_i = U_i * alpha;
            vec Z_tau_i = Z_i * tau;
            vec X_beta_i = X_i * beta;
            vec sigma_v_sq_i = exp(U_alpha_i);
            vec sqrt_sigma_v_sq_i = sqrt(sigma_v_sq_i);
            vec sigma_epsilon_sq_i = exp(Z_tau_i);

            double h_i = 0.0;
            
            // Numerical integration
            for (int q = 0; q < nq; ++q) {
                vec mu = X_beta_i + sqrt_sigma_v_sq_i * current_points(q);
                vec epsilon = y_i - mu;
                double log_lik_sum = sum(log(2 * M_PI * sigma_epsilon_sq_i) + (square(epsilon) / sigma_epsilon_sq_i));
                
                // Safe exponentiation
                likelihood_values(q) = exp(std::max(-0.5 * log_lik_sum, -700.0));
                h_i += likelihood_values(q) * current_weights(q);
            }

            // Update estimates
            if (h_i > std::numeric_limits<double>::min()) {
                double scal = 1.0 / h_i;
                double numerator_theta = sum(likelihood_values % current_points % current_weights);
                theta_new(i) = scal * numerator_theta;

                double numerator_var = sum(likelihood_values % square(current_points - theta_new(i)) % current_weights);
                psd_new(i) = sqrt(std::max(scal * numerator_var, 1e-12));
            } else {
                theta_new(i) = eb_prev_theta(i);
                psd_new(i) = eb_prev_psd(i);
            }
        }
    }
};

//' Update Empirical Bayes Estimates for Stage 1 & 2
//'
//' Calculates the updated Empirical Bayes (EB) estimates for the random effects
//' means (theta) and standard deviations (psd) using adaptive Gaussian quadrature.
//'
//' @param params A vector of model parameters (beta, alpha, tau).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y The response vector.
//' @param id A vector of subject IDs.
//' @param points Vector of standard Quadrature points.
//' @param weights Vector of standard Quadrature weights.
//' @param eb_prev_theta Vector of previous iteration's EB mean estimates.
//' @param eb_prev_psd Vector of previous iteration's EB standard deviation estimates.
//'
//' @return A list containing:
//' \item{eb_theta}{Updated vector of random effect means.}
//' \item{eb_psd}{Updated vector of random effect standard deviations.}
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List update_eb_estimates_rcpp(const arma::vec& params,
                                    const arma::mat& X, const arma::mat& U, const arma::mat& Z,
                                    const arma::vec& y, const arma::uvec& id,
                                    const arma::vec& points, const arma::vec& weights,
                                    const arma::vec& eb_prev_theta, const arma::vec& eb_prev_psd) {
    uvec unique_ids = unique(id);
    uword n_subjects = unique_ids.n_elem;
    
    // Pre-allocate output containers
    vec theta_new = zeros<vec>(n_subjects);
    vec psd_new = zeros<vec>(n_subjects);

    // Instantiate and run worker
    EBEstimatesWorker worker(params, X, U, Z, y, id, unique_ids, points, weights, 
                             eb_prev_theta, eb_prev_psd, theta_new, psd_new);
    parallelFor(0, n_subjects, worker);

    return Rcpp::List::create(Rcpp::Named("eb_theta") = theta_new,
                              Rcpp::Named("eb_sd") = psd_new);
}


// -----------------------------------------------------------------------------
// WORKER 2: Stage 3 Bivariate EB Estimates
// -----------------------------------------------------------------------------

struct Stage3EBEstimatesWorker : public Worker {
    // Inputs
    const vec& params;
    const mat& X;
    const mat& U;
    const mat& Z;
    const vec& y;
    const uvec& id;
    const uvec& unique_ids;
    const vec& points;
    const vec& weights;
    const std::string& model_type;
    const int adaptive;
    const mat& eb_prev_theta2;
    const mat& eb_prev_thetav;

    // Outputs
    mat& theta2_new;
    mat& thetav_new;

    // Internal
    int n_fixed, n_random_bs, n_random_ws, nq;

    // Constructor
    Stage3EBEstimatesWorker(const vec& params, const mat& X, const mat& U, const mat& Z,
                            const vec& y, const uvec& id, const uvec& unique_ids,
                            const vec& points, const vec& weights, 
                            const std::string& model_type, int adaptive,
                            const mat& eb_prev_theta2, const mat& eb_prev_thetav,
                            mat& theta2_new, mat& thetav_new)
        : params(params), X(X), U(U), Z(Z), y(y), id(id), unique_ids(unique_ids),
          points(points), weights(weights), model_type(model_type), adaptive(adaptive),
          eb_prev_theta2(eb_prev_theta2), eb_prev_thetav(eb_prev_thetav),
          theta2_new(theta2_new), thetav_new(thetav_new)
    {
        n_fixed = X.n_cols;
        n_random_bs = U.n_cols;
        n_random_ws = Z.n_cols;
        nq = points.n_elem;
    }

    void operator()(std::size_t begin, std::size_t end) {
        
        vec beta = params.subvec(0, n_fixed - 1);
        vec alpha = params.subvec(n_fixed, n_fixed + n_random_bs - 1);
        vec tau = params.subvec(n_fixed + n_random_bs, n_fixed + n_random_bs + n_random_ws - 1);
        vec sigma = params.subvec(n_fixed + n_random_bs + n_random_ws, params.n_elem - 1);

        // Pre-allocate buffers
        mat likelihood_values(nq, nq);
        vec theta_vec;
        if (model_type == "independent") { theta_vec.set_size(1); }
        else if (model_type == "linear") { theta_vec.set_size(2); }
        else { theta_vec.set_size(3); }

        for (std::size_t i = begin; i < end; ++i) {
            uvec idx = find(id == unique_ids(i));
            mat X_i = X.rows(idx);
            mat U_i = U.rows(idx);
            mat Z_i = Z.rows(idx);
            vec y_i = y.elem(idx);

            vec X_beta_i = X_i * beta;
            vec Z_tau_i = Z_i * tau;
            vec bsvar_i = exp(U_i * alpha);

            // Adaptive grid
            vec points1 = points, weights1 = weights;
            vec points2 = points, weights2 = weights;

            if (adaptive == 1 && i < eb_prev_theta2.n_rows) {
                if (eb_prev_thetav(i, 0) > 1e-8) {
                    double sd1 = sqrt(eb_prev_thetav(i, 0));
                    points1 = eb_prev_theta2(i, 0) + sd1 * points;
                    weights1 = weights * sd1 % exp(0.5 * (square(points) - square(points1)));
                }
                if (eb_prev_thetav(i, 2) > 1e-8) {
                    double sd2 = sqrt(eb_prev_thetav(i, 2));
                    points2 = eb_prev_theta2(i, 1) + sd2 * points;
                    weights2 = weights * sd2 % exp(0.5 * (square(points) - square(points2)));
                }
            }

            double h_i = 0.0;
            
            for (int q1 = 0; q1 < nq; ++q1) {
                double theta_1i = points1(q1);
                vec errij = y_i - (X_beta_i + sqrt(bsvar_i) * theta_1i);

                for (int q2 = 0; q2 < nq; ++q2) {
                    double theta_2i = points2(q2);
                    
                    if (model_type == "independent") { theta_vec(0) = theta_2i; }
                    else if (model_type == "linear") { theta_vec(0) = theta_1i; theta_vec(1) = theta_2i; }
                    else if (model_type == "quadratic") { theta_vec(0) = theta_1i; theta_vec(1) = theta_1i*theta_1i; theta_vec(2) = theta_2i; }
                    else { theta_vec(0) = theta_1i; theta_vec(1) = theta_1i*theta_2i; theta_vec(2) = theta_2i; }
                    
                    vec wsvar = exp(Z_tau_i + as_scalar(theta_vec.t() * sigma));
                    double log_prob_sum = sum(log(2 * M_PI * wsvar) + (square(errij) / wsvar));
                    double lik_val = exp(std::max(-0.5 * log_prob_sum, -700.0));
                    
                    likelihood_values(q1, q2) = lik_val;
                    h_i += lik_val * weights1(q1) * weights2(q2);
                }
            }

            if (h_i > 1e-100) {
                double scal = 1.0 / h_i;
                mat w_mat = weights1 * weights2.t(); 
                
                double mean1_num = accu(likelihood_values % repmat(points1, 1, nq) % w_mat);
                double mean2_num = accu(likelihood_values % repmat(points2.t(), nq, 1) % w_mat);
                
                // Write to distinct row i
                theta2_new(i, 0) = scal * mean1_num;
                theta2_new(i, 1) = scal * mean2_num;
                
                mat diff1_mat = repmat(points1 - theta2_new(i, 0), 1, nq);
                mat diff2_mat = repmat((points2 - theta2_new(i, 1)).t(), nq, 1);
                
                double var1_num = accu(likelihood_values % square(diff1_mat) % w_mat);
                double var2_num = accu(likelihood_values % square(diff2_mat) % w_mat);
                double cov_num  = accu(likelihood_values % (diff1_mat % diff2_mat) % w_mat);
                
                thetav_new(i, 0) = std::max(scal * var1_num, 1e-12);
                thetav_new(i, 1) = scal * cov_num;
                thetav_new(i, 2) = std::max(scal * var2_num, 1e-12);
            } else {
                theta2_new.row(i) = eb_prev_theta2.row(i);
                thetav_new.row(i) = eb_prev_thetav.row(i);
            }
        }
    }
};

//' Update Bivariate Empirical Bayes Estimates for Stage 3
//'
//' Updates the bivariate EB estimates for the mixed effects location-scale model.
//' Handles independent, linear, interaction, and quadratic model types.
//'
//' @param params A vector of model parameters (beta, alpha, tau, sigma).
//' @param X Design matrix for fixed effects.
//' @param U Design matrix for between-subject variance.
//' @param Z Design matrix for within-subject variance.
//' @param y The response vector.
//' @param id A vector of subject IDs.
//' @param points Vector of standard Quadrature points.
//' @param weights Vector of standard Quadrature weights.
//' @param model_type String specifying the model type ("independent", "linear", "quadratic" and "interaction").
//' @param adaptive Integer flag (0 or 1) indicating if adaptive quadrature is used.
//' @param eb_prev_theta2 Matrix (n x 2) of previous EB mean estimates.
//' @param eb_prev_thetav Matrix (n x 3) of previous EB variance-covariance estimates (var1, cov, var2).
//'
//' @return A list containing:
//' \item{theta2_eb}{Updated matrix (n x 2) of random effect means.}
//' \item{thetav_eb}{Updated matrix (n x 3) of variances and covariance.}
//' @keywords internal
// [[Rcpp::export]]
Rcpp::List update_eb_estimates_stage3_rcpp(const arma::vec& params,
                                           const arma::mat& X, const arma::mat& U, const arma::mat& Z,
                                           const arma::vec& y, const arma::uvec& id,
                                           const arma::vec& points, const arma::vec& weights,
                                           const std::string& model_type, int adaptive,
                                           const arma::mat& eb_prev_theta2, const arma::mat& eb_prev_thetav) {
    uvec unique_ids = unique(id);
    uword n_subjects = unique_ids.n_elem;
    
    // Pre-allocate output containers
    mat theta2_new = zeros<mat>(n_subjects, 2);
    mat thetav_new = zeros<mat>(n_subjects, 3);

    // Instantiate and run worker
    Stage3EBEstimatesWorker worker(params, X, U, Z, y, id, unique_ids, points, weights,
                                   model_type, adaptive, eb_prev_theta2, eb_prev_thetav,
                                   theta2_new, thetav_new);
    parallelFor(0, n_subjects, worker);

    return Rcpp::List::create(Rcpp::Named("theta2_eb") = theta2_new,
                              Rcpp::Named("thetav_eb") = thetav_new);
}